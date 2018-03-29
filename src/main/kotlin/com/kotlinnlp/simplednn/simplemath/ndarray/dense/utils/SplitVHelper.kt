package com.kotlinnlp.simplednn.simplemath.ndarray.dense.utils

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * An helper to split dense arrays that have been concatenated vertically.
 *
 * @param sizes the sizes of each array in which an input must be split
 */
class SplitVHelper(vararg sizes: Int) {

  /**
   * The array of sizes pf the split vectors.
   */
  private val sizeList: IntArray = sizes

  /**
   * The length that an input array must have to be split.
   */
  private val totalLength: Int = sizes.sum()

  /**
   * Split a given dense array in more vertical vectors, each with the related size given to the constructor of this
   * [SplitVHelper].
   *
   * @param array the input dense array
   *
   * @return a list of dense arrays
   */
  fun split(array: DenseNDArray): List<DenseNDArray> {

    require(array.isVector && array.length == this.totalLength) { "Array length not valid to be split"}

    return array.splitV(*this.sizeList).toList()
  }
}
