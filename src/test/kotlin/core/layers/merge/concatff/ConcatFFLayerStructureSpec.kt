/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain contexte at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.concatff

import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class ConcatFFLayerStructureSpec : Spek({

  describe("a ConcatLayer") {

    context("forward") {

      val layer = ConcatFFLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.079830, -0.669590, -0.777888)).equals(
            layer.outputArray.values,
            tolerance = 1.0e-06)
        }
      }
    }

    context("backward") {

      val layer = ConcatFFLayerUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(ConcatFFLayerUtils.getOutputErrors())

      val paramsErrors: ParamsErrorsList = layer.backward(propagateToInput = true)

      it("should match the expected errors of the weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.625985, -0.625985, -0.417323, -0.069554, 0.0, -0.34777, 0.486877, 0.486877, -0.556431),
            doubleArrayOf(0.397187, -0.397187, -0.264791, -0.044132, 0.0, -0.22066, 0.308923, 0.308923, -0.353055),
            doubleArrayOf(-0.213241, 0.213241, 0.14216, 0.023693, 0.0, 0.118467, -0.165854, -0.165854, 0.189547)
          )).equals(paramsErrors.getErrorsOf(layer.params.output.unit.weights)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the bias") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.695539, -0.441319, 0.236934))
            .equals(paramsErrors.getErrorsOf(layer.params.output.unit.biases)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the inputArray at index 0") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.029384, 0.109724, -0.259506, -0.573413)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the inputArray at index 1") {
        assertTrue {
          layer.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.104841, -0.234286)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the inputArray at index 2") {
        assertTrue {
          layer.inputArrays[2].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.628016, 0.295197, 0.751871)),
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
