/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain contexte at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.avg

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class AvgLayerStructureSpec : Spek({

  describe("a AvgLayer") {

    context("forward") {

      val layer = AvgLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.275, 0.075, 0.025)),
            tolerance = 1.0e-05)
        }
      }
    }

    context("backward") {

      val layer = AvgLayerUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(AvgLayerUtils.getOutputErrors())
      layer.backward(propagateToInput = true)

      it("should match the expected errors of the inputArray at index 0") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.25, -0.05, 0.1)),
            tolerance = 1.0e-05)
        }
      }

      it("should match the expected errors of the inputArray at index 1") {
        assertTrue {
          layer.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.25, -0.05, 0.1)),
            tolerance = 1.0e-05)
        }
      }

      it("should match the expected errors of the inputArray at index 2") {
        assertTrue {
          layer.inputArrays[2].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.25, -0.05, 0.1)),
            tolerance = 1.0e-05)
        }
      }

      it("should match the expected errors of the inputArray at index 3") {
        assertTrue {
          layer.inputArrays[3].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.25, -0.05, 0.1)),
            tolerance = 1.0e-05)
        }
      }
    }
  }
})
