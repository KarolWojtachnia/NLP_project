from datasets import load_dataset
import datasets as dt
import tensorflow_datasets as tfds

ds = tfds.load('huggingface:hate_speech_pl')