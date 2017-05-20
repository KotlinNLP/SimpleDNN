/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package logicgates

import com.kotlinnlp.simplednn.dataset.SimpleExample

fun main(args: Array<String>) {
  println("Start 'AND Gate Test'")
  println("Accuracy (softmax): %.1f%%".format(100.0 * ANDGate.testAccuracyWithSoftmax()))
  println("Accuracy (sigmoid): %.1f%%".format(100.0 * ANDGate.testAccuracyWithSigmoid()))
  println("End.")
}

object ANDGate {

  /**
   *
   */
  fun testAccuracyWithSoftmax(): Double {

    val examples: ArrayList<SimpleExample> = ArrayList()

    examples.addAll(listOf(
      SimpleExample(doubleArrayOf(0.0, 0.0), doubleArrayOf(1.0, 0.0)),
      SimpleExample(doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0, 0.0)),
      SimpleExample(doubleArrayOf(1.0, 0.0), doubleArrayOf(1.0, 0.0)),
      SimpleExample(doubleArrayOf(1.0, 1.0), doubleArrayOf(0.0, 1.0))
    ))

    return GateTestUtils.testAccuracyWithSoftmax(inputSize = 2, examples = examples, epochs = 1000)
  }

  /**
   *
   */
  fun testAccuracyWithSigmoid(): Double {

    val examples: ArrayList<SimpleExample> = ArrayList()

    examples.addAll(listOf(
      SimpleExample(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0)),
      SimpleExample(doubleArrayOf(0.0, 1.0), doubleArrayOf(0.0)),
      SimpleExample(doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0)),
      SimpleExample(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0))
    ))

    return GateTestUtils.testAccuracyWithSigmoid(inputSize = 2, examples = examples, epochs = 1000)
  }
}
