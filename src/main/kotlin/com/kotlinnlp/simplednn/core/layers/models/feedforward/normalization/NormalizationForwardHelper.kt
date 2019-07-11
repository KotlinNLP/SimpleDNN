package com.kotlinnlp.simplednn.core.layers.models.feedforward.normalization

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

class NormalizationForwardHelper <InputNDArrayType : NDArray<InputNDArrayType>>(
    override val layer: NormalizationLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  override fun forward() {


  }
}