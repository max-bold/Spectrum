from typing import Literal


RTANoiseGeneratorType = Literal["periodic IFFT", "filtered IIR"]
PERIODIC_IFFT_GENERATOR: RTANoiseGeneratorType = "periodic IFFT"
FILTERED_IIR_GENERATOR: RTANoiseGeneratorType = "filtered IIR"
RTA_NOISE_GENERATORS = (PERIODIC_IFFT_GENERATOR, FILTERED_IIR_GENERATOR)
