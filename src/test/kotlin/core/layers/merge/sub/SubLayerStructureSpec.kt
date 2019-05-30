/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.sub

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

class SubLayerStructureSpec : Spek({

  describe("a Subtract Layer") {

    context("forward") {

      val layer = SubLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.2, 0.3, -0.7)),
            tolerance = 1.0e-05)
        }
      }
    }

    context("backward") {

      val layer = SubLayerUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(SubLayerUtils.getOutputErrors())
      layer.backward(propagateToInput = true)

      it("should match the expected errors of the inputArray1") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4, 0.0)),
            tolerance = 1.0e-05)
        }
      }

      it("should match the expected errors of the inputArray2") {
        assertTrue {
          layer.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(1.0, 0.2, -0.4, 0.0)),
            tolerance = 1.0e-05)
        }
      }
    }
  }
})
