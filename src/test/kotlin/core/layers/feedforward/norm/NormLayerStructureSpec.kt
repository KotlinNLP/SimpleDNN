/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.norm

import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class NormLayerStructureSpec : Spek({

  describe("a NormLayer") {

    context("forward") {

      val layer = NormLayerStructureUtils.buildLayer()
      layer.forward()

      it("should match the expected output") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(1.15786, 0.2, -0.561559, -0.44465)),
            tolerance = 1.0e-06)
        }
      }
    }

    context("backward") {

      val layer = NormLayerStructureUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(NormLayerStructureUtils.getOutputErrors())

      val paramsErrors = layer.backward(propagateToInput = true)
      val params = layer.params

      it("should match the expected errors of the input") {
        assertTrue {
          layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.64465, 0.0, -0.193395, 0.77358)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the weights g") {
        assertTrue {
          paramsErrors.getErrorsOf(params.g)!!.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.64465, -0.25786, -0.451255, -0.483487)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the bias b") {
        assertTrue {
          paramsErrors.getErrorsOf(params.b)!!.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4, 0.6)),
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
