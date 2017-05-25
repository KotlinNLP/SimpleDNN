/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package layers.structure

import com.kotlinnlp.simplednn.core.layers.recurrent.gru.GRULayerParameters
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArrayFactory

import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import layers.structure.contextwindows.GRULayerContextWindow
import layers.structure.utils.GRULayerStructureUtils
import kotlin.test.assertEquals

/**
 *
 */
class GRULayerStructureSpec : Spek({

  describe("a GRULayerStructure") {

    context("forward") {

      on("without previous state context") {

        val layer = GRULayerStructureUtils.buildLayer(GRULayerContextWindow.Empty())
        layer.forward()

        it("should match the expected reset gate") {
          assertEquals(true, (layer.resetGate.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.40, 0.25, 0.50, 0.70, 0.45)),
            tolerance = 0.005))
        }

        it("should match the expected partition gate") {
          assertEquals(true, layer.partitionGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.85, 0.43, 0.12, 0.52, 0.24)),
            tolerance = 0.005))
        }

        it("should match the expected candidate") {
          assertEquals(true, layer.candidate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.87, -0.54, 0.96, 0.94, -0.21)),
            tolerance = 0.005))
        }

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.74, -0.23, 0.11, 0.49, -0.05)),
            tolerance = 0.005))
        }
      }

      on("with previous state context") {

        val layer = GRULayerStructureUtils.buildLayer(GRULayerContextWindow.Back())
        layer.forward()

        it("should match the expected reset gate") {
          assertEquals(true, layer.resetGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.72, 0.25, 0.55, 0.82, 0.53)),
            tolerance = 0.005))
        }

        it("should match the expected partition gate") {
          assertEquals(true, layer.partitionGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.91, 0.18, 0.05, 0.67, 0.39)),
            tolerance = 0.005))
        }

        it("should match the expected candidate") {
          assertEquals(true, layer.candidate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.96, 0.07, 0.92, 0.97, 0.33)),
            tolerance = 0.005))
        }

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.86, 0.18, -0.24, 0.36, -0.36)),
            tolerance = 0.005))
        }
      }
    }

    context("backward") {

      on("without previous and next state") {

        val layer = GRULayerStructureUtils.buildLayer(GRULayerContextWindow.Empty())
        val paramsErrors = GRULayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = GRULayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.17, -0.98, 0.26, -1.15, -0.5)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate") {
          assertEquals(true, layer.resetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate") {
          assertEquals(true, layer.partitionGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.02, 0.13, 0.03, -0.27, 0.02)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.04, -0.3, 0.0, -0.07, -0.12)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate biases") {
          assertEquals(true, paramsErrors.resetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate biases") {
          assertEquals(true, paramsErrors.partitionGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.02, 0.13, 0.03, -0.27, 0.02)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.04, -0.3, 0.0, -0.07, -0.12)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate weights") {
          assertEquals(true, (paramsErrors.resetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate weights") {
          assertEquals(true, (paramsErrors.partitionGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.01, -0.02, -0.02, 0.02),
              doubleArrayOf(-0.10, -0.12, -0.12, 0.13),
              doubleArrayOf(-0.02, -0.02, -0.02, 0.03),
              doubleArrayOf(0.22, 0.24, 0.24, -0.27),
              doubleArrayOf(-0.02, -0.02, -0.02, 0.02)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.03, -0.03, -0.03, 0.04),
              doubleArrayOf(0.24, 0.27, 0.27, -0.30),
              doubleArrayOf(0.00, 0.00, 0.00, 0.00),
              doubleArrayOf(0.06, 0.06, 0.06, -0.07),
              doubleArrayOf(0.09, 0.10, 0.10, -0.12)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate recurrent weights") {
          assertEquals(true, paramsErrors.resetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate recurrent weights") {
          assertEquals(true, paramsErrors.partitionGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate recurrent weights") {
          assertEquals(true, paramsErrors.candidate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.53, -0.49, 0.18, 0.20)),
            tolerance = 0.005))
        }
      }

      on("with previous state only") {

        val layer = GRULayerStructureUtils.buildLayer(GRULayerContextWindow.Back())
        val paramsErrors = GRULayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = GRULayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.29, -0.57, -0.09, -1.28, -0.81)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate") {
          assertEquals(true, layer.resetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.01, 0.0, -0.02, -0.03, -0.01)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate") {
          assertEquals(true, layer.partitionGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.03, 0.01, -0.01, -0.52, -0.22)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.02, -0.10, 0.00, -0.06, -0.28)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate biases") {
          assertEquals(true, paramsErrors.resetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.01, 0.0, -0.02, -0.03, -0.01)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate biases") {
          assertEquals(true, paramsErrors.partitionGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.03, 0.01, -0.01, -0.52, -0.22)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.02, -0.10, 0.00, -0.06, -0.28)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate weights") {
          assertEquals(true, (paramsErrors.resetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.00, 0.01, 0.01, -0.01),
              doubleArrayOf(0.00, 0.00, 0.00, 0.00),
              doubleArrayOf(0.01, 0.01, 0.01, -0.02),
              doubleArrayOf(0.02, 0.02, 0.02, -0.03),
              doubleArrayOf(0.01, 0.01, 0.01, -0.01)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate weights") {
          assertEquals(true, (paramsErrors.partitionGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.02, -0.02, -0.02, 0.03),
              doubleArrayOf(-0.01, -0.01, -0.01, 0.01),
              doubleArrayOf(0.0, 0.0, 0.0, -0.01),
              doubleArrayOf(0.42, 0.47, 0.47, -0.52),
              doubleArrayOf(0.17, 0.2, 0.2, -0.22)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.02, -0.02, -0.02, 0.02),
              doubleArrayOf(0.08, 0.09, 0.09, -0.10),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.05, 0.05, 0.05, -0.06),
              doubleArrayOf(0.22, 0.25, 0.25, -0.28)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate recurrent weights") {
          assertEquals(true, paramsErrors.resetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.01, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.01, 0.01),
              doubleArrayOf(0.01, -0.01, 0.01, 0.02, 0.02),
              doubleArrayOf(0.0, 0.0, 0.0, 0.01, 0.01)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate recurrent weights") {
          assertEquals(true, paramsErrors.partitionGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.01, 0.01, -0.01, -0.02, -0.02),
              doubleArrayOf(0.0, 0.0, 0.0, -0.01, -0.01),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.1, -0.1, 0.16, 0.47, 0.42),
              doubleArrayOf(0.04, -0.04, 0.07, 0.2, 0.17)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate recurrent weights") {
          assertEquals(true, paramsErrors.candidate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, -0.01, -0.01),
              doubleArrayOf(0.01, -0.01, 0.02, 0.08, 0.04),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.01, 0.0, 0.01, 0.04, 0.02),
              doubleArrayOf(0.04, -0.01, 0.05, 0.21, 0.12)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.56, -0.83, 0.50, 0.55)),
            tolerance = 0.005))
        }
      }

      on("with next state only") {

        val layer = GRULayerStructureUtils.buildLayer(GRULayerContextWindow.Front())
        val paramsErrors = GRULayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = GRULayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.05, -0.24, 0.94, -0.18, -0.71)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate") {
          assertEquals(true, layer.resetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate") {
          assertEquals(true, layer.partitionGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.01, 0.03, 0.09, -0.04, 0.03)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.01, -0.07, 0.01, -0.01, -0.17)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate biases") {
          assertEquals(true, paramsErrors.resetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate biases") {
          assertEquals(true, paramsErrors.partitionGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.01, 0.03, 0.09, -0.04, 0.03)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.01, -0.07, 0.01, -0.01, -0.17)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate weights") {
          assertEquals(true, (paramsErrors.resetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate weights") {
          assertEquals(true, (paramsErrors.partitionGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.00, 0.00, 0.00, 0.01),
              doubleArrayOf(-0.03, -0.03, -0.03, 0.03),
              doubleArrayOf(-0.07, -0.08, -0.08, 0.09),
              doubleArrayOf(0.03, 0.04, 0.04, -0.04),
              doubleArrayOf(-0.02, -0.02, -0.02, 0.03)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.01, -0.01, -0.01, 0.01),
              doubleArrayOf(0.06, 0.07, 0.07, -0.07),
              doubleArrayOf(-0.01, -0.01, -0.01, 0.01),
              doubleArrayOf(0.01, 0.01, 0.01, -0.01),
              doubleArrayOf(0.13, 0.15, 0.15, -0.17)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate recurrent weights") {
          assertEquals(true, paramsErrors.resetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate recurrent weights") {
          assertEquals(true, paramsErrors.partitionGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate recurrent weights") {
          assertEquals(true, paramsErrors.candidate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.08, -0.13, 0.11, 0.12)),
            tolerance = 0.005))
        }
      }

      on("with previous and next state") {

        val layer = GRULayerStructureUtils.buildLayer(GRULayerContextWindow.Bilateral())
        val paramsErrors = GRULayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = GRULayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.17, 0.17, 0.58, -0.31, -1.02)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate") {
          assertEquals(true, layer.resetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.01, 0.0, -0.01, -0.03, 0.01)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate") {
          assertEquals(true, layer.partitionGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.02, 0.0, 0.03, -0.13, -0.28)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.01, 0.03, 0.0, -0.01, -0.35)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate biases") {
          assertEquals(true, paramsErrors.resetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.01, 0.0, -0.01, -0.03, 0.01)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate biases") {
          assertEquals(true, paramsErrors.partitionGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.02, 0.0, 0.03, -0.13, -0.28)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.01, 0.03, 0.0, -0.01, -0.35)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate weights") {
          assertEquals(true, (paramsErrors.resetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.01, 0.01, 0.01, -0.01),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.01, 0.01, 0.01, -0.01),
              doubleArrayOf(0.02, 0.02, 0.02, -0.03),
              doubleArrayOf(-0.01, -0.01, -0.01, 0.01)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate weights") {
          assertEquals(true, (paramsErrors.partitionGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.01, -0.01, -0.01, 0.02),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(-0.03, -0.03, -0.03, 0.03),
              doubleArrayOf(0.10, 0.11, 0.11, -0.13),
              doubleArrayOf(0.22, 0.25, 0.25, -0.28)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.01, -0.01, -0.01, 0.01),
              doubleArrayOf(-0.02, -0.03, -0.03, 0.03),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.01, 0.01, 0.01, -0.01),
              doubleArrayOf(0.28, 0.32, 0.32, -0.35)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the reset gate recurrent weights") {
          assertEquals(true, paramsErrors.resetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.01, 0.01),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.01, 0.01),
              doubleArrayOf(0.01, -0.01, 0.01, 0.02, 0.02),
              doubleArrayOf(0.0, 0.0, 0.0, -0.01, -0.01)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the partition gate recurrent weights") {
          assertEquals(true, paramsErrors.partitionGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, -0.01, -0.01),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(-0.01, 0.01, -0.01, -0.03, -0.03),
              doubleArrayOf(0.03, -0.03, 0.04, 0.11, 0.10),
              doubleArrayOf(0.06, -0.06, 0.08, 0.25, 0.22)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate recurrent weights") {
          assertEquals(true, paramsErrors.candidate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, -0.01, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, -0.02, -0.01),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.01, 0.01),
              doubleArrayOf(0.05, -0.02, 0.06, 0.26, 0.15)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.11, -0.46, 0.46, 0.53)),
            tolerance = 0.005))
        }
      }
    }
  }
})
