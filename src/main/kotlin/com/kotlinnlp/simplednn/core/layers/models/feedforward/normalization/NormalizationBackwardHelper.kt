package com.kotlinnlp.simplednn.core.layers.models.feedforward.normalization

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

class NormalizationBackwardHelper <InputNDArrayType : NDArray<InputNDArrayType>>(
    override val layer: NormalizationLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  override fun execBackward(propagateToInput: Boolean) {
  this.layer.applyOutputActivationDerivs()
    for ((index, outputArray) in this.layer.outputArrays.withIndex()){
      val gy: DenseNDArray = outputArray.errors

      this.layer.params.b.errors.values.assignSum(gy)
      val sub : DenseNDArray = DenseNDArrayFactory.zeros(this.layer.inputArrays[0].values.shape)
      sub.assignValues(this.layer.inputArrays[index].values)
      sub.assignSub(this.layer.meanArray).assignDiv(this.layer.devStdArray.assignSum(0.00000000001))

      this.layer.params.g.errors.values.assignSum(sub.assignProd(gy))
      if (propagateToInput) {
        this.layer.inputArrays[index].assignErrors(gy.assignProd(this.layer.params.g.values.div(this.layer.devStdArray)))
      }
    }
  }
}