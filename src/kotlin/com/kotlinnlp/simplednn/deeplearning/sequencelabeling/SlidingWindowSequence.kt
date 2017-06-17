/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.sequencelabeling

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A simple implementation of a Sliding Window Sequence (SWS).
 *
 * SlidingWindowSequence implements convenient method to process a sequence,
 * viewed as an array of consecutive [elements].
 * This class is useful when you need to see an element 'e' along with its left and right context 'c'.
 *
 * The base case has the form of c-3 c-2 c-1 e c+1 c+2 c+3.
 *
 * So, every element is once treated as a focus element, and when the window moves, it becomes a context element.
 * When the focus is the 1st element in the sequence, there are no context elements on the left,
 * when it's the 2nd element, there's only one context element on the left, and so on.
 *
 * @property elements the dense representation of the sequence elements
 * @property leftContextSize the number of elements used to create the left context
 * @property rightContextSize the number of elements used to create the right context
 */
class SlidingWindowSequence(
  val elements: Array<DenseNDArray>,
  val leftContextSize: Int = 3,
  val rightContextSize: Int = 3) {

  /**
   * The index of the focus element [-1 .. sequence length]
   */
  var focusIndex: Int = -1
    private set

  /**
   * @return the number of the elements of the sequence
   */
  val size get() = this.elements.size

  /**
   * @param index of an element within the sequence
   *
   * @return returns the element of the sequence at the specified [index]
   */
  operator fun get(index: Int) = this.elements[index]

  /**
   * Set the focus at the specified [index]
   *
   * @param index of an element within the sequence
   */
  fun setFocus(index: Int){
    require(index in -1 .. this.elements.size - 1)

    this.focusIndex = index
  }

  /**
   * @return the dense-representation of the focus element
   */
  fun getFocusElement(): DenseNDArray = this.elements[this.focusIndex]

  /**
   *
   * @return False if the focus element is the last element of the sequence, True otherwise
   */
  fun hasNext(): Boolean = this.focusIndex + 1 < this.elements.size

  /**
   * Shift the focus by one position
   */
  fun shift() {
    require(this.hasNext()) {
      "The focus element [${this.focusIndex}] is the last element of the sequence."
    }

    this.focusIndex++
  }

  /**
   * @return an array containing the indexes of the elements within the [leftContextSize] or 'null' values if an index
   *         is out of the range of the sequence.
   */
  fun getLeftContext(): Array<Int?> = Array(
    size = this.leftContextSize,
    init = {
      val i = this.leftContextSize - 1 - it
      val k = this.focusIndex - 1 - i
      if (k < 0) null else k
    }
  )

  /**
   * @return an array containing the indexes of the elements within the [rightContextSize] or 'null' values if an index
   *         is out of the range of the sequence.
   */
  fun getRightContext(): Array<Int?> = Array(
    size = this.rightContextSize,
    init = {
      val k = this.focusIndex + 1 + it
      if (k >= this.elements.size) null else k
    }
  )

  /**
   * @return the entire focused context composed by the indexes of the left context, the focus and the right context
   */
  fun getContext(): Array<Int?> = this.getLeftContext() + this.focusIndex + this.getRightContext()

  /**
   * @return a string representation of the focused context
   */
  fun contextToString(): String =
    "[${this.getLeftContext().joinToString()}] $focusIndex [${this.getRightContext().joinToString()}]"
}
