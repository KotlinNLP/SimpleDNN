/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.affine

import com.kotlinnlp.simplednn.core.layers.models.merge.affine.AffineLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */
class AffineLayerStructureSpec : Spek({

  describe("an AffineLayerStructure") {

    on("forward") {

      val layer = AffineLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.664037, -0.019997)),
            tolerance = 1.0e-06)
        }
      }
    }

    on("backward") {

      val layer = AffineLayerUtils.buildLayer()
      val paramsErrors = AffineLayerParameters(inputsSize = listOf(2, 3), outputSize = 2)

      layer.forward()

      layer.outputArray.assignErrors(layer.outputArray.values.sub(AffineLayerUtils.getOutputGold()))
      layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

      it("should match the expected errors of the outputArray") {
        assertTrue {
          layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.427139, 0.279891)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the biases") {
        assertTrue {
          paramsErrors.b.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.427139, 0.279891)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of w1") {
        assertTrue {
          (paramsErrors.w[0].values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.341711, 0.384425),
              doubleArrayOf(-0.223913, -0.251902)
            )),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of w2") {
        assertTrue {
          (paramsErrors.w[1].values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.213569, 0.085428, -0.256283),
              doubleArrayOf(0.139945, -0.055978, 0.167934)
            )),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the inputArray1") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.095771, -0.537634)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the inputArray2") {
        assertTrue {
          layer.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.172316, -0.297537, 0.468392)),
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
