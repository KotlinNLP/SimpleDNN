/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.biaffine

import com.kotlinnlp.simplednn.core.layers.merge.biaffine.BiaffineLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */
class BiaffineLayerStructureSpec : Spek({

  describe("a BiaffineLayerStructure") {

    on("forward") {

      val layer = BiaffineLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertEquals(true, layer.outputArray.values.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.714345, -0.161572)),
          tolerance = 1.0e-06))
      }
    }

    on("backward") {

      val layer = BiaffineLayerUtils.buildLayer()
      val paramsErrors = BiaffineLayerParameters(inputSize1 = 2, inputSize2 = 3, outputSize = 2)

      layer.forward()

      layer.outputArray.assignErrors(layer.outputArray.values.sub(BiaffineLayerUtils.getOutputGold()))
      layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

      it("should match the expected errors of the outputArray") {
        assertEquals(true, layer.outputArray.errors.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.398794, 0.134815)),
          tolerance = 1.0e-06))
      }

      it("should match the expected errors of the biases") {
        assertEquals(true, paramsErrors.b.values.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.398794, 0.134815)),
          tolerance = 1.0e-06))
      }

      it("should match the expected errors of w1") {
        assertEquals(true, (paramsErrors.w1.values as DenseNDArray).equals(
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.319035, 0.358915),
			      doubleArrayOf(-0.107852, -0.121333)
          )),
          tolerance = 1.0e-06))
      }

      it("should match the expected errors of w2") {
        assertEquals(true, (paramsErrors.w2.values as DenseNDArray).equals(
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.199397, 0.079759, -0.239276),
            doubleArrayOf(0.067407, -0.026963, 0.080889)
          )),
          tolerance = 1.0e-06))
      }

      it("should match the expected errors of the first w array") {
        assertEquals(true, (paramsErrors.w[0].values as DenseNDArray).equals(
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.159518, 0.179457),
            doubleArrayOf(-0.063807, -0.071783),
            doubleArrayOf(0.191421, 0.215349)
          )),
          tolerance = 1.0e-06))
      }

      it("should match the expected errors of the second w array") {
        assertEquals(true, (paramsErrors.w[1].values as DenseNDArray).equals(
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.053926, -0.060667),
            doubleArrayOf(0.021570, 0.024267),
            doubleArrayOf(-0.064711, -0.072800)
          )),
          tolerance = 1.0e-06))
      }

      it("should match the expected errors of the inputArray1") {
        assertEquals(true, layer.inputArray1.errors.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.048872, -0.488442)),
          tolerance = 1.0e-06))
      }

      it("should match the expected errors of the inputArray2") {
        assertEquals(true, layer.inputArray2.errors.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.342293, -0.086394, 0.601735)),
          tolerance = 1.0e-06))
      }
    }
  }
})
