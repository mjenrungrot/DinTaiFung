from typing import List

import numpy as np
import librosa
import soundfile as sf

from constants import \
    SPEED_OF_SOUND, ATTENUATION_ALPHA

INPUT_OUTPUT_TARGET_SAMPLE_RATE: int = 48000


class Microphone(object):
    """
    Microphone class
    """
    def __init__(self, position: List[float]):
        """
        Args:
            position: x,y,z list of floats
        """
        self.position = np.array(position)
        self.buffer = np.array([])
        self.sources_gt = []
        self.sample_rate = None

    def save(self, output_prefix: str):
        """
        Write mixed buffer and gt sources
        """
        sf.write(output_prefix + "mixed.wav", self.buffer, self.sample_rate)
        for idx, source in enumerate(self.sources_gt):
            sf.write(output_prefix + "source{:02}_gt.wav".format(idx), source,
                     self.sample_rate)

    def reset(self):
        """
        Empty all buffers
        """
        self.buffer = np.array([])
        self.sources_gt = []
        self.sample_rate = None


class SoundSource(object):
    """
    SoundSource class
    """
    def __init__(self,
                 position: List[float],
                 filename=None,
                 data=None,
                 sr=None,
                 offset: float = 0.0,
                 duration=None,
                 start_time: float = 0.0,
                 trim_silence=True,
                 reduce_factor=1.0):
        """
        Either filename should be passed, or data and sample rate
        """
        assert len(position) == 3  # x, y, z
        self.position = np.array(position)

        if filename is None:
            if data is None or sr is None:
                raise (ValueError(
                    "Either filename or audio and sampe rate must be provided")
                       )
            self.audio = data
            self.sample_rate = sr
            self.start_time = start_time

        elif isinstance(filename, str):
            audio, sample_rate = librosa.core.load(
                filename,
                sr=INPUT_OUTPUT_TARGET_SAMPLE_RATE,
                mono=True,
                offset=offset,
                duration=duration)
            self.audio = audio
            self.sample_rate = sample_rate
            self.start_time = start_time

        else:
            audio = np.array([])
            for f in filename:
                curr_audio, sample_rate = librosa.core.load(
                    f,
                    sr=INPUT_OUTPUT_TARGET_SAMPLE_RATE,
                    mono=True,
                    offset=offset,
                    duration=duration)
                audio = np.concatenate((audio, curr_audio))

            self.audio = audio
            self.sample_rate = sample_rate
            self.start_time = start_time
        # Trim silence at the beginning, making sure we have something
        if trim_silence:
            new_audio = librosa.effects.trim(self.audio, top_db=20)[0]
            if len(new_audio) < 100:
                new_audio = librosa.effects.trim(self.audio, top_db=80)[0]
            self.audio = new_audio

        self.audio *= reduce_factor

    def save(self, filename: str):
        sf.write(filename, self.audio, self.sample_rate)

class Scene(object):
    """
    Scene class
    """
    def __init__(self, sources: List[SoundSource], mics: List[Microphone]):
        self.sources = sources
        self.mics = mics

        assert all([x.sample_rate == sources[0].sample_rate for x in sources]), \
            "Sample rate conversion not supported right now"

        self.sample_rate = sources[0].sample_rate

    def render(self,
               cutoff_time: float,
               geometric_attenuation=True,
               atmospheric_attenuation=True,
               volume_boost=1.0):
        """
        Render all sound sources to all microphones.
        Only does ITD and attenuation.

        cutoff_time: in seconds
        """
        for mic in self.mics:
            # Initialize mic buffer to be empty correct size
            total_samples = int(cutoff_time * self.sample_rate)
            mic.buffer = np.zeros((total_samples))
            mic.sample_rate = self.sample_rate

            # Render all sources to that mic
            for source in self.sources:
                # ITD
                distance = np.linalg.norm(mic.position - source.position)
                curr_start_time = source.start_time + distance / SPEED_OF_SOUND
                curr_start_samples = int(source.sample_rate * curr_start_time)
                curr_buffer = np.concatenate((np.zeros(
                    (curr_start_samples)).astype(np.float64), source.audio))

                # Attenuation
                if geometric_attenuation:
                    curr_buffer = np.divide(
                        curr_buffer, distance**
                        2)  # attenuation due to energy spreading over area
                if atmospheric_attenuation:
                    curr_buffer = np.multiply(
                        curr_buffer, np.exp(-ATTENUATION_ALPHA * distance)
                    )  # attenuation due to atmosphere https://en.wikibooks.org/wiki/Engineering_Acoustics/Outdoor_Sound_Propagation

                # Pad if necessary to cutoff_time
                curr_buffer = np.pad(curr_buffer, (0, total_samples))
                curr_buffer = curr_buffer[:total_samples]
                curr_buffer *= volume_boost

                # Append to each mic
                mic.sources_gt.append(curr_buffer)
                mic.buffer += np.array(curr_buffer)

    def render_binaural(self, mic_idxs: List[int], output_filename: str,
                        cutoff_time: float):
        """
        Renders a stero output with two chosen mics
        """
        self.render(cutoff_time)
        left_data = self.mics[mic_idxs[0]].buffer
        right_data = self.mics[mic_idxs[1]].buffer
        left_data = np.expand_dims(left_data, 0)
        right_data = np.expand_dims(right_data, 0)

        stereo_data = np.concatenate((left_data, right_data))
        sf.write(output_filename, np.swapaxes(stereo_data, 0, 1),
                 self.sample_rate)
