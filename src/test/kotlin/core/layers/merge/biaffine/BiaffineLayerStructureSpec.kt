/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain contexte at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.biaffine

import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class BiaffineLayerStructureSpec : Spek({

  describe("a BiaffineLayer") {

    context("forward") {

      val layer = BiaffineLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.714345, -0.161572)),
            tolerance = 1.0e-06)
        }
      }
    }

    context("backward") {

      val layer = BiaffineLayerUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(layer.outputArray.values.sub(BiaffineLayerUtils.getOutputGold()))
      val paramsErrors = layer.backward(propagateToInput = true)

      val params = layer.params

      it("should match the expected errors of the outputArray") {
        assertTrue {
          layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.398794, 0.134815)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the biases") {
        assertTrue {
          paramsErrors.getErrorsOf(params.b)!!.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.398794, 0.134815)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of w1") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.w1)!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.319035, 0.358915),
              doubleArrayOf(-0.107852, -0.121333)
            )),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of w2") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.w2)!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.199397, 0.079759, -0.239276),
              doubleArrayOf(0.067407, -0.026963, 0.080889)
            )),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the first w array") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.w[0])!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.159518, 0.179457),
              doubleArrayOf(-0.063807, -0.071783),
              doubleArrayOf(0.191421, 0.215349)
            )),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the second w array") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.w[1])!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.053926, -0.060667),
              doubleArrayOf(0.021570, 0.024267),
              doubleArrayOf(-0.064711, -0.072800)
            )),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the inputArray1") {
        assertTrue {
          layer.inputArray1.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.048872, -0.488442)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the inputArray2") {
        assertTrue {
          layer.inputArray2.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.342293, -0.086394, 0.601735)),
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
