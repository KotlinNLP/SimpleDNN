/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.normalization

import com.kotlinnlp.simplednn.core.layers.models.feedforward.highway.HighwayLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import core.layers.feedforward.highway.HighwayLayerStructureUtils
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

class NormalizationLayerStructureSpec : Spek({

  describe("a Normalization") {

    context("forward") {

      val layer = NormalizationLayerStructureUtils.buildLayer()
      layer.forward()

      it("should match the expected output at position 0") {
        assertTrue {
          layer.outputArrays[0].values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.39693, -0.796878, 0.0, 0.701374)),
              tolerance = 1.0e-06)
        }
      }

      it("should match the expected output at position 1") {
        assertTrue {
          layer.outputArrays[0].values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.39693, -0.796878, 0.0, 0.701374)),
              tolerance = 1.0e-06)
        }
      }

      it("should match the expected output at position 2") {
        assertTrue {
          layer.outputArrays[0].values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.39693, -0.796878, 0.0, 0.701374)),
              tolerance = 1.0e-06)
        }
      }
    }

    context("backward") {

      val layer = NormalizationLayerStructureUtils.buildLayer()

      layer.forward()

      layer.outputArrays[0].assignErrors(NormalizationLayerStructureUtils.getOutputErrors1())
      layer.outputArrays[1].assignErrors(NormalizationLayerStructureUtils.getOutputErrors2())
      layer.outputArrays[2].assignErrors(NormalizationLayerStructureUtils.getOutputErrors3())

      val paramsErrors = layer.backward(propagateToInput = true)
      val params = layer.params

      it("should match the expected errors of the outputArray") {
        assertTrue {
          layer.outputArray.errors.equals(
              HighwayLayerStructureUtils.getOutputErrors(),
              tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the input 1") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.409706, 0.118504, -0.017413, 0.433277)),
              tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the input 2") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.409706, 0.118504, -0.017413, 0.433277)),
              tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the input 3") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.409706, 0.118504, -0.017413, 0.433277)),
              tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the bias b") {
        assertTrue {
          paramsErrors.getErrorsOf(params.b)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.409706, 0.118504, -0.017413, 0.433277)),
              tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the weights g") {
        assertTrue {
          paramsErrors.getErrorsOf(params.g)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.028775, 0.018987, -0.013853, -0.122241)),
              tolerance = 1.0e-06)
        }
      }
    }
  }
})
