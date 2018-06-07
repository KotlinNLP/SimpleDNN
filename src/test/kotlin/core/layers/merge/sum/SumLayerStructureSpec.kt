/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.sum

import com.kotlinnlp.simplednn.core.layers.merge.sum.SumLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */
class SumLayerStructureSpec : Spek({

  describe("a SumLayerStructure") {

    on("forward") {

      val layer = SumLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertEquals(true, layer.outputArray.values.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.1, 0.3, 0.1)),
          tolerance = 1.0e-05))
      }
    }

    on("backward") {

      val layer = SumLayerUtils.buildLayer()
      val paramsErrors = SumLayerParameters(inputSize = 3, nInputs = 4)

      layer.forward()

      layer.outputArray.assignErrors(SumLayerUtils.getOutputErrors())
      layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

      it("should match the expected errors of the inputArray at index 0") {
        assertEquals(true, layer.inputArrays[0].errors.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4)),
          tolerance = 1.0e-05))
      }

      it("should match the expected errors of the inputArray at index 1") {
        assertEquals(true, layer.inputArrays[1].errors.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4)),
          tolerance = 1.0e-05))
      }

      it("should match the expected errors of the inputArray at index 2") {
        assertEquals(true, layer.inputArrays[2].errors.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4)),
          tolerance = 1.0e-05))
      }

      it("should match the expected errors of the inputArray at index 3") {
        assertEquals(true, layer.inputArrays[3].errors.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4)),
          tolerance = 1.0e-05))
      }
    }
  }
})
