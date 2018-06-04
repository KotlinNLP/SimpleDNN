/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.structure

import com.kotlinnlp.simplednn.core.layers.feedforward.highway.HighwayLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import core.layers.structure.utils.HighwayLayerStructureUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */
class HighwayLayerStructureSpec : Spek({

  describe("a HighwayLayerStructure") {

    on("forward") {

      val layer = HighwayLayerStructureUtils.buildLayer()
      layer.forward()

      it("should match the expected input unit") {
        assertTrue {
          layer.inputUnit.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.39693, -0.796878, 0.0, 0.701374)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected transform gate") {
        assertTrue {
          layer.transformGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.85321, 0.432907, 0.116089, 0.519989)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.456097, -0.855358, -0.79552, 0.844718)),
            tolerance = 1.0e-06)
        }
      }
    }

    on("backward") {

      val layer = HighwayLayerStructureUtils.buildLayer()
      val paramsErrors = HighwayLayerParameters(inputSize = 4, outputSize = 4)

      layer.forward()

      layer.outputArray.assignErrors(HighwayLayerStructureUtils.getOutputErrors())
      layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

      it("should match the expected errors of the outputArray") {
        assertTrue {
          layer.outputArray.errors.equals(
            HighwayLayerStructureUtils.getOutputErrors(),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the input unit") {
        assertTrue {
          layer.inputUnit.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.409706, 0.118504, -0.017413, 0.433277)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the transform gate") {
        assertTrue {
          layer.transformGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.028775, 0.018987, -0.013853, -0.122241)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the input unit biases") {
        assertTrue {
          paramsErrors.input.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.409706, 0.118504, -0.017413, 0.433277)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the transform gate biases") {
        assertTrue {
          paramsErrors.transformGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.028775, 0.018987, -0.013853, -0.122241)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the input unit weights") {
        assertTrue {
          (paramsErrors.input.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.327765, -0.368736, -0.368736, 0.409706),
              doubleArrayOf(-0.094803, -0.106653, -0.106653, 0.118504),
              doubleArrayOf(0.013931, 0.015672, 0.015672, -0.017413),
              doubleArrayOf(-0.346622, -0.389949, -0.389949, 0.433277)
            )),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the transform gate weights") {
        assertTrue {
          (paramsErrors.transformGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.023020, -0.025897, -0.025897, 0.028775),
              doubleArrayOf(-0.015190, -0.017088, -0.017088, 0.018987),
              doubleArrayOf(0.011082, 0.012467, 0.012467, -0.013853),
              doubleArrayOf(0.097793, 0.110017, 0.110017, -0.122241)
            )),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the inputArray") {
        assertTrue {
          layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.705908, 0.245982, -0.453725, 0.394556)),
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
