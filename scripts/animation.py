import threading
from utils import slerp, remap, convert_to_index_dict, get_neighbors_and_weight
from attrdict import AttrDict
from image_classes import SDImageList, SDImageListList

animation_context = threading.local()


class BaseScheduler:

    def __call__(self, timestep):
        raise NotImplementedError

    def apply(self, timesteps):
        return [self(timestep) for timestep in timesteps]

    def description(self, timestep):
        return f"{self.__class__.__name__}({timestep})"



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
        print(before, after, weight)
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
        if not self._pre_processed:
            for key, val in self.embeddings.items():
                self.embeddings[key] = self.preprocess_fn(val)
            self._pre_processed = True
        before, after, weight = get_neighbors_and_weight(self.embeddings, timestep)
        slerped = slerp(weight, before, after)
        if self.postprocess_fn is not None:
            slerped = self.postprocess_fn(slerped)
        return slerped


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
        animation_context.runner = self
        frame_options = AttrDict(self.options.copy())
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
