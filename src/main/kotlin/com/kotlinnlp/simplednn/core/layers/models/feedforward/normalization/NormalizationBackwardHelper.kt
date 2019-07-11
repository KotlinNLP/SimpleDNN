package com.kotlinnlp.simplednn.core.layers.models.feedforward.normalization

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

class NormalizationBackwardHelper <InputNDArrayType : NDArray<InputNDArrayType>>(
    override val layer: NormalizationLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  override fun execBackward(propagateToInput: Boolean) {

  }
}