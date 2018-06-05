/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.mergelayers.product

import com.kotlinnlp.simplednn.core.mergelayers.product.ProductLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */
class ProductLayerStructureSpec : Spek({

  describe("a ProductLayerStructure") {

    context("4 input arrays") {

      on("forward") {

        val layer = ProductLayerUtils.buildLayer4()
        layer.forward()

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.126, 0.192)),
            tolerance = 1.0e-05))
        }
      }

      on("backward") {

        val layer = ProductLayerUtils.buildLayer4()
        val paramsErrors = ProductLayerParameters(inputSize = 3, nInputs = 4)

        layer.forward()

        layer.outputArray.assignErrors(ProductLayerUtils.getOutputErrors4())
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

        it("should match the expected errors of the inputArray at index 0") {
          assertEquals(true, layer.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.028, 0.128)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray at index 1") {
          assertEquals(true, layer.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.315, -0.0504, -0.1536)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray at index 2") {
          assertEquals(true, layer.inputArrays[2].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.036, 0.096)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray at index 3") {
          assertEquals(true, layer.inputArrays[3].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.063, -0.096)),
            tolerance = 1.0e-05))
        }
      }
    }

    context("5 input arrays") {

      on("forward") {

        val layer = ProductLayerUtils.buildLayer5()
        layer.forward()

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.1134, -0.096)),
            tolerance = 1.0e-05))
        }
      }

      on("backward") {

        val layer = ProductLayerUtils.buildLayer5()
        val paramsErrors = ProductLayerParameters(inputSize = 3, nInputs = 5)

        layer.forward()

        layer.outputArray.assignErrors(ProductLayerUtils.getOutputErrors5())
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

        it("should match the expected errors of the inputArray at index 0") {
          assertEquals(true, layer.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.063, 0.128)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray at index 1") {
          assertEquals(true, layer.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.11025, 0.1134, -0.1536)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray at index 2") {
          assertEquals(true, layer.inputArrays[2].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.081, 0.096)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray at index 3") {
          assertEquals(true, layer.inputArrays[3].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.14175, -0.096)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray at index 4") {
          assertEquals(true, layer.inputArrays[4].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.063, -0.1536)),
            tolerance = 1.0e-05))
        }
      }
    }
  }
})
