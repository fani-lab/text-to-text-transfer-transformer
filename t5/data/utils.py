# Copyright 2022 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for data loading and processing."""

import gin
import seqio

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100

#--gin_param="SentencePieceVocabulary.sentencepiece_model_file = './output/t5/vocabs/cc_all.32000/sentencepiece.model'"  ^
#--gin_param="tsv_dataset_fn.vocabulary = @SentencePieceVocabulary()" ^ ==> Does not work since seqio does not import gin and this functiontion does not have gin.configurable decorator
#--gin_param="run.vocabulary = @SentencePieceVocabulary()" ==> even if works, at t5.models.mesh_transformer_main.py#181, it is explicitly bind to a function. So, useless gin_param
# Either hard code vocab path or below solution

@gin.configurable
def get_default_spm_path(path=None):
    global DEFAULT_SPM_PATH
    return path if path else DEFAULT_SPM_PATH

#t5.data.tasks module calls this function way before gin params init for creating the tasks/mixtures.
#So, the runner will raise exception if a specific task is requested.
#it's now ignored by --module_import="numpy" or any common module (default is t5.data.mixtures) and MIXTURE_NAME='' (empty) in --gin_param
def get_default_vocabulary():
    return seqio.SentencePieceVocabulary(get_default_spm_path(), DEFAULT_EXTRA_IDS)

# ========================= Mixing Rate Functions ==============================


@gin.configurable
def rate_num_examples(
    task, maximum=None, temperature=1.0, scale=1.0,
    fallback_to_num_input_examples=True):
  """Mixing rate equal to the number of examples for the task."""
  return seqio.mixing_rate_num_examples(
      task=task, maximum=maximum, scale=scale, temperature=temperature,
      fallback_to_num_input_examples=fallback_to_num_input_examples)


@gin.configurable
def rate_unsupervised(task, value=1e6):
  """Gin-configurable mixing rate for the unsupervised co-training task."""
  del task
  return value
