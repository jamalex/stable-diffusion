from contextlib import contextmanager
from functools import reduce
import threading
from utils import slerp, remap, convert_to_index_dict, get_neighbors_and_weight, find_indices
from attrdict import AttrDict
from image_classes import SDImage, SDImageList, SDImageListList

animation_context = threading.local()


@contextmanager
def set_animation_context(**kwargs):
    animation_context.__dict__.update(kwargs)
    yield
    animation_context.__dict__.clear()


def convert_to_latent(frame):
    return animation_context.runner.controller.encode_to_torch(frame)


class BaseScheduler:

    def __call__(self, timestep):
        raise NotImplementedError

    def apply(self, timesteps):
        return [self(timestep) for timestep in timesteps]

    def description(self, timestep):
        return f"{self.__class__.__name__}({timestep})"

    def __add__(self, other):
        if isinstance(other, BaseScheduler):
            return ComposeSchedulers([self, other], reduce_fn=lambda x, y: x + y)
        elif isinstance(other, (int, float)):
            return Postprocess(self, lambda x: x + other if x is not None else None)

    def __mul__(self, other):
        if isinstance(other, BaseScheduler):
            return ComposeSchedulers([self, other], reduce_fn=lambda x, y: x * y)
        elif isinstance(other, (int, float)):
            return Postprocess(self, lambda x: x * other if x is not None else None)

    def __truediv__(self, other):
        if isinstance(other, BaseScheduler):
            return ComposeSchedulers([self, other], reduce_fn=lambda x, y: x / y)
        elif isinstance(other, (int, float)):
            return Postprocess(self, lambda x: x / other if x is not None else None)

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return Postprocess(self, lambda x: -x)

    def __abs__(self):
        return Postprocess(self, lambda x: x.abs() if hasattr(x, "abs") else abs(x))


class ComposeSchedulers(BaseScheduler):

    def __init__(self, schedulers, reduce_fn, remove_none=True):
        self.schedulers = schedulers
        self.reduce_fn = reduce_fn
        self.remove_none = remove_none

    def __call__(self, timestep):
        input_results = [scheduler(timestep) for scheduler in self.schedulers]
        if self.remove_none:
            input_results = [x for x in input_results if x is not None]
        if input_results:
            return reduce(self.reduce_fn, input_results)
        else:
            return None


class Postprocess(BaseScheduler):
    
        def __init__(self, scheduler, postprocess_fn):
            self.scheduler = scheduler
            self.postprocess_fn = postprocess_fn
    
        def __call__(self, timestep):
            result = self.scheduler(timestep)
            if result is None:
                return None
            else:
                return self.postprocess_fn(result)


class Conditional(BaseScheduler):

    def __init__(self, condition_fn, true_scheduler, false_scheduler=None):
        self.condition_fn = condition_fn
        self.true_scheduler = true_scheduler
        self.false_scheduler = false_scheduler

    def __call__(self, timestep):
        if self.condition_fn(timestep):
            if self.true_scheduler is not None:
                return self.true_scheduler(timestep)
        else:
            if self.false_scheduler is not None:
                return self.false_scheduler(timestep)


class ExponentialFloat(BaseScheduler):
    """
    Scheduler that does exponential interpolation between start and end, with specified exponent.
    Exponents > 1 will make the changes in value start slow and end up fast, and vice versa.
    """

    def __init__(self, start, end, exponent):
        self.start = start
        self.end = end
        self.exponent = exponent

    def __call__(self, timestep):
        return remap(timestep ** self.exponent, 0, 1, self.start, self.end)
        

class CombineNearestFrames(BaseScheduler):
    """
    Scheduler that merges the nearest two frames into a single frame using the specified merge_fn.
    (the merge_fn should take a dict mapping relative position to two frames, and return a single frame)
    """

    def __init__(self, merge_fn):
        self.merge_fn = merge_fn

    def __call__(self, timestep):
        print("COMBINING NEAREST FRAMES")
        frames = animation_context.runner.get_closest_generated_frames(timestep)
        if len(frames) == 1:
            return convert_to_latent(list(frames.values())[0])
        else:
            return self.merge_fn(frames)


class AverageOfNeighborFrames(CombineNearestFrames):
    """
    Scheduler that returns the average of the nearest frames. If a weight_fn is specified, it
    will be used to weight the frames (0=all the before frame, 1=all the after frame).
    """

    def __init__(self, weight_fn=lambda d: 0.5):
        
        def get_average_frame(relframedict):
            weight = weight_fn(relframedict)
            if len(relframedict) == 0:
                return None
            elif len(relframedict) == 1:
                if list(relframedict.keys())[0] < 0 and weight == 0:
                    return convert_to_latent(list(relframedict.values())[0])
                elif list(relframedict.keys())[0] > 0 and weight == 1:
                    return convert_to_latent(list(relframedict.values())[0])
                else:
                    return None
            elif len(relframedict) == 2:
                before_key, after_key = list(relframedict.keys())
                assert before_key < 0 and after_key > 0
                before_val, after_val = [convert_to_latent(frame) for frame in relframedict.values()]
                if weight == 0:
                    return before_val
                elif weight == 1:
                    return after_val
                else:
                    return before_val * (1 - weight) + after_val * weight
            else:
                raise ValueError("relframedict should have at most two elements")
        
        super().__init__(get_average_frame)


class WeightedAverageOfNeighborFrames(AverageOfNeighborFrames):
    """
    Scheduler that returns the weighted average of the nearest frames. The weight_curve_fn
    converts a linear distance into the actual weight to use (e.g. could be exponential).
    """

    def __init__(self, weight_curve_fn=lambda w: w):
        def get_weight(relframedict):
            if len(relframedict) == 0:
                return 0.5
            elif len(relframedict) == 1:
                return 0 if list(relframedict.keys())[0] < 0 else 1
            elif len(relframedict) == 2:
                before_key, after_key = list(relframedict.keys())
                assert before_key < 0 and after_key > 0
                return weight_curve_fn(abs(after_key) / (abs(before_key) + abs(after_key)))
            else:
                raise ValueError("relframedict should have at most two elements")
        super().__init__(weight_fn=get_weight)


class PreviousFrame(AverageOfNeighborFrames):
    """
    Scheduler that returns the previous frame.
    """

    def __init__(self):
        super().__init__(weight_fn=lambda d: 0)


class NextFrame(AverageOfNeighborFrames):
    """
    Scheduler that returns the next frame.
    """

    def __init__(self):
        super().__init__(weight_fn=lambda d: 1)


class NearestInSequence(BaseScheduler):
    """
    Scheduler that returns the nearest item (mapping the timestep to the indices) in a sequence.
    """

    def __init__(self, values):
        self.values = convert_to_index_dict(values)

    def __call__(self, timestep):
        before, after, weight = get_neighbors_and_weight(self.values, timestep)
        return before if weight < 0.5 else after


class LinearInterpolation(BaseScheduler):
    """
    Scheduler that does linear interpolation between the two nearest items in a sequence
    (after mapping the timestep to the indices to find the neighbors).
    """
    
    def __init__(self, values):
        self.values = convert_to_index_dict(values)

    def __call__(self, timestep):
        before, after, weight = get_neighbors_and_weight(self.values, timestep)
        return (1 - weight) * before + weight * after


class Slerp(BaseScheduler):
    """
    Scheduler that performs weighted slerp (spherical linear interpolation) between embeddings.
    """

    _pre_processed = False

    def __init__(self, embeddings, preprocess_fn=None, postprocess_fn=None):
        self.embeddings = convert_to_index_dict(embeddings)
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

    def __call__(self, timestep):
        if not self._pre_processed and self.preprocess_fn is not None:
            for key, val in self.embeddings.items():
                self.embeddings[key] = self.preprocess_fn(val)
            self._pre_processed = True
        before, after, weight = get_neighbors_and_weight(self.embeddings, timestep)
        slerped = slerp(weight, before, after)
        if self.postprocess_fn is not None:
            slerped = self.postprocess_fn(slerped)
        return slerped


class SlerpImageLatents(Slerp):
    """
    Scheduler that performs weighted slerp (spherical linear interpolation) between latent representations of images.
    """

    def __init__(self, images, preprocess_fn=None, postprocess_fn=None):
        def proxy_preprocess_fn(x):
            if preprocess_fn is not None:
                x = preprocess_fn(x)
            if isinstance(x, SDImage):
                x = animation_context.runner.controller.encode_to_torch(x)
            return x
        super().__init__(images, proxy_preprocess_fn, postprocess_fn)


class SlerpPrompts(Slerp):
    """
    Convenience scheduler that takes raw string prompts and encodes them before slerping.
    """

    def __init__(self, prompts, preprocess_fn=None, postprocess_fn=None):
        def proxy_preprocess_fn(x):
            if preprocess_fn is not None:
                x = preprocess_fn(x)
            if isinstance(x, str):
                x = animation_context.runner.controller.encode_prompt(x)
            return x
        super().__init__(prompts, preprocess_fn=proxy_preprocess_fn, postprocess_fn=postprocess_fn)



class AnimationRunner:

    def __init__(self, name, controller, prompt, verbose=True, **options):
        animation_context.runner = self
        self.name = name
        self.controller = controller
        self.options = AttrDict(options.copy())
        self.options.n_iter = 1  # don't run more than one batch
        self.options.prompt = prompt
        self.results = {}
        self.verbose = verbose

    def calculate_frame_options(self, timestep):
        """
        Calculates the options for the frame at the specified timestep.
        """
        frame_options = AttrDict(self.options.copy())
        with set_animation_context(runner=self, frame_options=frame_options):
            for key, value in self.options.items():
                # if the value is a scheduler, calculate its value for this timestep
                if isinstance(value, BaseScheduler):
                    frame_options[key] = value(timestep)
                    if self.verbose:
                        print(f"{key} = {frame_options[key]} ({value.description(timestep)})")
        return frame_options

    def get_frame(self, timestep, index=0, generate_if_needed=True):
        """
        Returns the frame at the specified timestep, running the model if needed to get it.
        The index is used to reference the target sample within a multi-sample batch.
        """
        if timestep not in self.results:
            if generate_if_needed:
                self.generate(timestep)
            else:
                return None
        images = self.results[timestep]["images"]
        if index is None:
            return images
        elif isinstance(index, (int, slice)):
            return images[index]
        else:
            raise ValueError(f"Invalid index: {index}")

    def get_frames(self, timesteps, index=0):
        """
        Returns the frames at the specified timesteps, running the model if needed to get them.
        """
        frames = []
        for i, timestep in enumerate(timesteps):
            frames.append(self.get_frame(timestep, index=index))
            if self.verbose:
                print("Got frame %10f (%d of %d)" % (timestep, i + 1, len(timesteps)))
        if isinstance(index, int):
            return SDImageList(frames)
        else:
            return SDImageListList(frames)

    def get_linspace_frames(self, start, end, num, index=0):
        """
        Returns the frames at the specified number of evenly spaced timesteps between start and end.
        """
        timesteps = [start + (end - start) * i / (num - 1) for i in range(num)]
        return self.get_frames(timesteps, index=index)

    def get_generated_frames(self, start=0, end=1, index=0):
        """
        Returns all the frames that have already been generated, between start and end timesteps.
        """
        return self.get_frames([key for key in sorted(self.results.keys()) if start <= key <= end], index=index)

    def get_relative_frames_dict(self, reference_timestep, timesteps, index=0, generate_if_needed=False):
        """
        Returns a dictionary of frames relative to the specified reference timestep.
        """
        frames = {}
        if isinstance(timesteps, (int, float)):
            timesteps = [timesteps]
        for timestep in sorted(timesteps):
            frame = self.get_frame(timestep, index=index, generate_if_needed=generate_if_needed)
            if frame is not None:
                frames[timestep - reference_timestep] = frame
        return frames

    def get_closest_generated_frames(self, timestep, index=0):
        """
        Returns the closest generated frames to the specified timestep, as a relative frames dict.
        """
        keys = list(sorted(self.results.keys()))
        if len(keys) == 0:
            return {}
        if timestep <= keys[0]:
            return self.get_relative_frames_dict(timestep, keys[0])
        elif timestep >= keys[-1]:
            return self.get_relative_frames_dict(timestep, keys[-1])
        else:
            before_i, after_i = find_indices(timestep, keys)
            before, after = keys[before_i], keys[after_i]
            return self.get_relative_frames_dict(timestep, [before, after], index=index)

    def generate(self, timestep):
        """
        Generates the frame at the specified timestep.
        """
        options = self.calculate_frame_options(timestep)
        self.results[timestep] = self.controller.generate(**options)
        for index, image in enumerate(self.results[timestep]["images"]):
            image.update_metadata(
                timestep=timestep,
                index=index,
                animation_name=self.name,
                **options,
            )
        return self.results[timestep]
