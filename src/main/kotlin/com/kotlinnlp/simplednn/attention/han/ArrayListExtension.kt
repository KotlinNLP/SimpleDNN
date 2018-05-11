package com.kotlinnlp.simplednn.attention.han

import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * @return a [HierarchySequence] from an array of [DenseNDArray]s
 */
fun <NDArrayType: NDArray<NDArrayType>>Array<NDArrayType>.toHierarchySequence(): HierarchySequence<NDArrayType> {
  return HierarchySequence(*this)
}
