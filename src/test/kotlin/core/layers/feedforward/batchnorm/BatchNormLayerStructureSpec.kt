/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.batchnorm

import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class BatchNormLayerStructureSpec : Spek({

  describe("a BatchNormLayer") {

    context("forward") {

      val layer = BatchNormLayerStructureUtils.buildLayer()
      layer.forward()

      it("should match the expected output at position 0") {
        assertTrue {
          layer.outputArrays[0].values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(1.182833, 0.2, -0.519764, -0.130704)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected output at position 1") {
        assertTrue {
          layer.outputArrays[1].values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.334334, 0.2, -0.92716, -0.571642)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected output at position 2") {
        assertTrue {
          layer.outputArrays[2].values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(1.182833, 0.2, -1.253076, 1.302346)),
            tolerance = 1.0e-06)
        }
      }
    }

    context("backward") {

      val layer = BatchNormLayerStructureUtils.buildLayer()

      layer.forward()

      layer.outputArrays[0].assignErrors(BatchNormLayerStructureUtils.getOutputErrors1())
      layer.outputArrays[1].assignErrors(BatchNormLayerStructureUtils.getOutputErrors2())
      layer.outputArrays[2].assignErrors(BatchNormLayerStructureUtils.getOutputErrors3())

      val paramsErrors = layer.backward(propagateToInput = true)
      val params = layer.params

      it("should match the expected errors of the input at position 0") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.060623, 0.0, -0.325917, 0.661408)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the input at position 1") {
        assertTrue {
          layer.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.318187, 0.0, -0.570354, 0.992111)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the input at position 2") {
        assertTrue {
          layer.inputArrays[2].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.318187, 0.0, -0.570354, -0.881877)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the weights g") {
        assertTrue {
          paramsErrors.getErrorsOf(params.g)!!.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.070708, -0.475549, 0.380236, -2.218471)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the bias b") {
        assertTrue {
          paramsErrors.getErrorsOf(params.b)!!.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.5, 1.8, 0.7)),
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
