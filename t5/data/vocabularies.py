# Copyright 2020 The T5 Authors.
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

# Lint as: python3
"""Abstract Vocabulary."""

import abc
from typing import Iterable, Optional, Sequence

import tensorflow.compat.v2 as tf


PAD_ID = 0
EOS_ID = 1
UNK_ID = 2


class Vocabulary(metaclass=abc.ABCMeta):
  """Abstract class for all vocabularies.

  Subclasses must implement methods for converting between strings and tokens
  both in pure python (`_encode`/`_decode`) and in TensorFlow
  (`_encode_tf`/`_decode_tf`).

  Subclasses are responsible for reserving PAD_ID=0 as well as EOS_ID and UNK_ID
  if `use_eos` and `use_unk` are True, respectively.

  `_base_vocab_size` should account for PAD, EOS, and UNK but not `extra_ids`.
  """

  def __init__(
      self,
      extra_ids: int = 0,
      use_eos: bool = True,
      use_unk: bool = True):
    """Vocabulary constructor.

    Args:
      extra_ids: The number of extra IDs to reserve.
      use_eos: Whether to stop decoding at EOS_ID=1.
      use_unk: Whether to replace tokens out of range with UNK_ID=2.
    """
    self._extra_ids = extra_ids
    self._use_eos = use_eos
    self._use_unk = use_unk

  @property
  def eos_id(self) -> Optional[int]:
    return EOS_ID if self._use_eos else None

  @property
  def pad_id(self) -> int:
    return PAD_ID

  @property
  def unk_id(self) -> Optional[int]:
    return UNK_ID if self._use_unk else None

  @property
  def extra_ids(self) -> int:
    return self._extra_ids

  @property
  def vocab_size(self) -> int:
    """Vocabulary size, including extra ids."""
    return self._base_vocab_size + self.extra_ids

  @abc.abstractproperty
  def _base_vocab_size(self) -> int:
    """Vocabulary size, excluding extra ids but including PAD/EOS/UNK."""
    raise NotImplementedError

  @abc.abstractmethod
  def _encode(self, s: str) -> Sequence[int]:
    raise NotImplementedError

  def encode(self, s: str) -> Sequence[int]:
    """Tokenizes string to an int sequence, without adding EOS."""
    return self._encode(s)

  @abc.abstractmethod
  def _decode(self, ids):
    raise NotImplementedError

  def decode(self, ids: Iterable[int]):
    """Detokenizes int32 iterable to a string, up to EOS."""
    clean_ids = list(ids)

    if self.unk_id is not None:
      clean_ids = [
          self.unk_id if i >= self._base_vocab_size else i
          for i in clean_ids
      ]

    if self.eos_id is not None and self.eos_id in clean_ids:
      clean_ids = clean_ids[:clean_ids.index(self.eos_id)]

    return self._decode(clean_ids)

  @abc.abstractmethod
  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError

  def encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    """Tokenizes string Scalar to an int32 Tensor, without adding EOS."""
    return self._encode_tf(s)

  @abc.abstractmethod
  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError

  def decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    """Detokenizes int32 Tensor to a string Scalar, up to EOS."""
    valid_ids = tf.constant(ids)

    if self.unk_id is not None:
      valid_ids = tf.where(
          tf.less(valid_ids, self._base_vocab_size), valid_ids, self.unk_id)

    if self.eos_id is not None:
      # Argmax always returns the first occurrence.
      first_eos = tf.argmax(tf.equal(valid_ids, self.eos_id))
      valid_ids = tf.cond(
          tf.logical_and(
              tf.equal(first_eos, 0),
              tf.not_equal(valid_ids[0], self.eos_id)),
          lambda: valid_ids,
          lambda: valid_ids[:first_eos])

    return self._decode_tf(valid_ids)


class ByteVocabulary(Vocabulary):
  """Byte level vocabulary.

  Build mappings between Unicode characters and IDs. Encode/decode
  Unicode characeters/IDs based on UTF-8. Reserve ID=0 is for padding,
  ID=1 for EOS, and ID=2 for UNK.
  """

  def __init__(self, extra_ids: int = 0):
    """Create a ByteVocabulary.

    Optionally, specify a number of extra ids to add to the end of the
    vocabulary for use as sentinels.

    Args:
      extra_ids: an optional integer
    """
    self._byte_size = 256
    # The special tokens: 0=PAD, 1=EOS,and 2=UNK
    self._num_special_tokens = 3
    super().__init__(use_eos=True, use_unk=True, extra_ids=extra_ids)

  def _convert_strings_to_ids(self, s):
    """Convert a python string to integers based on UTF-8 encoding.

    Args:
      s: a string
    Returns:
      ids: a list of integers
    """
    return list(s.encode("utf-8"))

  def _convert_ids_to_strings(self, ids):
    """Convert ids to a python string based on UTF-8 encoding.

    Args:
      ids: a list of integers
    Returns:
      s: a string
    """
    return bytes(ids).decode("utf-8", errors="ignore")

  def _filter_non_string_ids(self, ids):
    """Filter special token ids and extra ids if there are any.

    Args:
      ids: a list of integers
    Returns:
      ids: a list of integers
    """
    lower_bound = self._num_special_tokens
    upper_bound = self._byte_size + self._num_special_tokens
    return [id for id in ids if lower_bound <= id < upper_bound]

  @property
  def _base_vocab_size(self):
    """Number of ids.

    Returns:
      an integer, the vocabulary size
    """
    return self._num_special_tokens + self._byte_size

  def _encode(self, s):
    """Encode a python string as a list of integers.

    To keep the first few ids for special tokens, increase ids by the number
    of special tokens.

    Args:
      s: a string
    Returns:
      a list of integers (not terminated by EOS)
    """
    ids = self._convert_strings_to_ids(s)
    return [i + self._num_special_tokens for i in ids]

  def _decode(self, ids):
    """Decode a list of integers to a python string.

    The special tokens of PAD, EOS, and UNK will not be represented in the
    output string. This is different from the sentencepiece_vocabulary, where
    UNK will show up as a '?' character.

    Args:
      ids: a list of integers (not terminated by EOS)
    Returns:
      a string
    """

    ids = self._filter_non_string_ids(ids)
    ids = [i - self._num_special_tokens for i in ids]
    return self._convert_ids_to_strings(ids)

  def _encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

    Args:
      s: a tf.Scalar with dtype tf.string
    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    tf_ids = tf.io.decode_raw(s, tf.uint8) + self._num_special_tokens
    return tf.dtypes.cast(tf_ids, tf.int32)

  def _decode_tf(self, ids):
    """Decode in TensorFlow.

    Args:
      ids: a 1d tf.Tensor with dtype tf.int32
    Returns:
      a tf Scalar with dtype tf.string
    """
    return tf.py_function(func=self.decode, inp=[ids], Tout=tf.string)

  def __eq__(self, other):
    their_extra_ids = other.extra_ids
    return self.extra_ids == their_extra_ids
