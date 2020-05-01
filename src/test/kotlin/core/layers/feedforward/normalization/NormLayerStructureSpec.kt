/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.normalization

import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

class NormLayerStructureSpec : Spek({

  describe("a NormLayer") {

    context("forward") {

      val layer = NormLayerStructureUtils.buildLayer()
      layer.forward()

      it("should match the expected output at position 0") {
        assertTrue {
          layer.outputArrays[0].values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(1.1828427, 0.2, 0.0, 0.0)),
              tolerance = 1.0e-06)
        }
      }

      it("should match the expected output at position 1") {
        assertTrue {
          layer.outputArrays[1].values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.334314, 0.2, 0.0, 0.0)),
              tolerance = 1.0e-06)
        }
      }

      it("should match the expected output at position 2") {
        assertTrue {
          layer.outputArrays[2].values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(1.1828427, 0.2, 0.0, 1.302356)),
              tolerance = 1.0e-06)
        }
      }
    }

    context("backward") {

      val layer = NormLayerStructureUtils.buildLayer()

      layer.forward()

      layer.outputArrays[0].assignErrors(NormLayerStructureUtils.getOutputErrors1())
      layer.outputArrays[1].assignErrors(NormLayerStructureUtils.getOutputErrors2())
      layer.outputArrays[2].assignErrors(NormLayerStructureUtils.getOutputErrors3())

      val paramsErrors = layer.backward(propagateToInput = true)
      val params = layer.params

      it("should match the expected errors of the input 1") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.060660,	0.0,	0.0,	0.0)),
              tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the input 2") {
        assertTrue {
          layer.inputArrays[1].errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.318198,	0.0,	0.0,	0.0)),
              tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the input 3") {
        assertTrue {
          layer.inputArrays[2].errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.318198,	0.0,	0.0,	-0.881885)),
              tolerance = 1.0e-06)

        }
      }

      it("should match the expected errors of the bias b") {
        assertTrue {
          paramsErrors.getErrorsOf(params.b)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.5, 0.0, -0.8)),
              tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the weights g") {
        assertTrue {
          paramsErrors.getErrorsOf(params.g)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.070710, -0.475556, 0.0, -1.102356)),
              tolerance = 1.0e-06)
        }
      }
    }
  }
})
