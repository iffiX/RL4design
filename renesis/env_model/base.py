from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    The base environment model
    """

    def __init__(self):
        self.steps = 0

    @property
    @abstractmethod
    def action_space(self):
        """Returns the space of action."""
        return None

    @property
    @abstractmethod
    def observation_space(self):
        """Returns the space of observation."""
        return None

    @abstractmethod
    def reset(self):
        """Reset model"""
        raise NotImplementedError()

    @abstractmethod
    def is_finished(self):
        """Returns True if there is no more changes that can be applied."""
        raise NotImplementedError()

    @abstractmethod
    def is_robot_invalid(self):
        """
        Returns True if robot is invalid (no voxels, not continuous, etc.)
        Environment may use this to set the fitness (reward) score.
        """
        raise NotImplementedError()

    def step(self, action):
        """Change the robot design"""
        raise NotImplementedError()

    def observe(self):
        """
        Observe and get view
        """
        raise NotImplementedError()

    def get_robot(self):
        """
        Returns the current robot for creating a voxcraft simulation. Different
        environments may support different representations, some only returns
        material, some others return amplitude, frequency, etc.

        Returns:
            sizes:
            A tuple of size containing (x, y, z)

            representation:
            A list of tuples of length z (voxel max height), each tuple is of form
            (material, amplitude, frequency, phase shift), and each element in tuple
            is a list of length x*y, where x and y are the bounding box sizes
            of all voxels.
        """
        raise NotImplementedError()

    def get_robot_voxels(self):
        """
        Returns:
            A three-dimensional numpy array of shape
            [dimension_size, dimension_size, dimension_size]
            and dtype np.float32 specifying the voxel placement,
            0 corresponds to empty.

        Note:
            This function only return voxels belonging to the robot we simulate,
            other voxels are set to 0
        """
        raise NotImplementedError()

    def get_voxels(self):
        """
        Returns:
            A three dimensional numpy array of shape
            [dimension_size, dimension_size, dimension_size]
            and dtype np.float32 specifying the voxel placement,
            0 corresponds to empty.
        """
        raise NotImplementedError()

    def get_state_data(self):
        """
        Returns any additional state data to be saved with robots
        """
        return None


class BaseVectorizedModel(ABC):
    """
    The base vectorized model, all sub env models have the same
    number of steps
    """

    def __init__(self):
        self.steps = 0
        self.env_num = 0

    @property
    @abstractmethod
    def action_space(self):
        """Returns the space of action."""
        return None

    @property
    @abstractmethod
    def observation_space(self):
        """Returns the space of observation."""
        return None

    @property
    @abstractmethod
    def initial_observation_after_reset_single_env(self):
        """Returns the initial observation after resetting a single sub env."""
        return None

    @abstractmethod
    def reset(self):
        """Reset model"""
        raise NotImplementedError()

    @abstractmethod
    def is_finished(self):
        """Returns True if there is no more changes that can be applied.
        I.e. return whether step number has reached maximum"""
        raise NotImplementedError()

    def step(self, actions):
        """Change robot designs"""
        raise NotImplementedError()

    def observe(self):
        """
        Observe and get view of all robots
        """
        raise NotImplementedError()

    def get_robots(self):
        """
        Returns the current robot for creating a voxcraft simulation. Different
        environments may support different representations, some only returns
        material, some others return amplitude, frequency, etc.

        Returns:
            sizes:
            A tuple of size containing (x, y, z)

            representation:
            A list of tuples of length z (voxel max height), each tuple is of form
            (material, amplitude, frequency, phase shift), and each element in tuple
            is a list of length x*y, where x and y are the bounding box sizes
            of all voxels.
        """
        raise NotImplementedError()

    def get_robots_voxels(self):
        """
        Returns:
            A three dimensional numpy array of shape
            [dimension_size, dimension_size, dimension_size]
            and dtype np.float32 specifying the voxel placement,
            0 corresponds to empty.

        Note:
            This function only return voxels belonging to the robot we simulate,
            other voxels are set to 0
        """
        raise NotImplementedError()

    def get_voxels(self):
        """
        Returns:
            A three dimensional numpy array of shape
            [robot_num, dimension_size, dimension_size, dimension_size]
            and dtype np.float32 specifying the voxel placement,
            0 corresponds to empty.
        """
        raise NotImplementedError()
