/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package layers.structure

import com.kotlinnlp.simplednn.core.layers.recurrent.ran.RANLayerParameters
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import layers.structure.contextwindows.RANLayerContextWindow
import layers.structure.utils.RANLayerStructureUtils
import kotlin.test.assertEquals

/**
 *
 */
class RANLayerStructureSpec : Spek({

  describe("a RANLayerStructure") {

    context("forward") {

      on("without previous state context") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayerContextWindow.Empty())
        layer.forward()

        it("should match the expected input gate") {
          assertEquals(true, layer.inputGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.39652, 0.25162, 0.5, 0.70475, 0.45264)),
            tolerance = 1.0e-05))
        }

        it("should match the expected forget gate") {
          assertEquals(true, layer.forgetGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.85321, 0.43291, 0.11609, 0.51999, 0.24232)),
            tolerance = 1.0e-05))
        }

        it("should match the expected candidate") {
          assertEquals(true, layer.candidate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(1.02, -0.1, 0.1, 2.03, -1.41)),
            tolerance = 1.0e-05))
        }

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.38375, -0.02516, 0.04996, 0.8918, -0.56369)),
            tolerance = 1.0e-05))
        }
      }

      on("with previous state context") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayerContextWindow.Back())
        layer.forward()

        it("should match the expected input gate") {
          assertEquals(true, layer.inputGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.72312, 0.24974, 0.54983, 0.82054, 0.53494)),
            tolerance = 1.0e-05))
        }

        it("should match the expected forget gate") {
          assertEquals(true, layer.forgetGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.91133, 0.18094, 0.04834, 0.67481, 0.38936)),
            tolerance = 1.0e-05))
        }

        it("should match the expected candidate") {
          assertEquals(true, layer.candidate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(1.02, -0.1, 0.1, 2.03, -1.41)),
            tolerance = 1.0e-05))
        }

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5045, 0.01121, 0.04046, 0.78504, -0.78786)),
            tolerance = 1.0e-05))
        }
      }
    }

    context("backward") {

      on("without previous and next state") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayerContextWindow.Empty())
        val paramsErrors = RANLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = RANLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.18625, -0.77516, 0.19996, -0.7482, -1.01369)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.04546, 0.0146, 0.005, -0.31604, 0.35412)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.07385, -0.19504, 0.09998, -0.52729, -0.45884)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.04546, 0.0146, 0.005, -0.31604, 0.35412)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.07385, -0.19504, 0.09998, -0.52729, -0.45884)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.03637, 0.04091, 0.04091, -0.04546),
              doubleArrayOf(-0.01168, -0.01314, -0.01314, 0.0146),
              doubleArrayOf(-0.004, -0.0045, -0.0045, 0.005),
              doubleArrayOf(0.25283, 0.28444, 0.28444, -0.31604),
              doubleArrayOf(-0.2833, -0.31871, -0.31871, 0.35412)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.05908, 0.06647, 0.06647, -0.07385),
              doubleArrayOf(0.15603, 0.17554, 0.17554, -0.19504),
              doubleArrayOf(-0.07998, -0.08998, -0.08998, 0.09998),
              doubleArrayOf(0.42183, 0.47456, 0.47456, -0.52729),
              doubleArrayOf(0.36707, 0.41296, 0.41296, -0.45884)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.41805, 0.33996, -0.21165, 0.39196)),
            tolerance = 1.0e-05))
        }
      }

      on("with previous state only") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayerContextWindow.Back())
        val paramsErrors = RANLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = RANLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.06550, -0.73879, 0.19046, -0.85496, -1.23786)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.01338, 0.01384, 0.00471, -0.25557, 0.43421)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.00104, -0.02161, -0.00255, 0.13439, 0.19543)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.04737, -0.18450, 0.10472, -0.70153, -0.66218)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.01338, 0.01384, 0.00471, -0.25557, 0.43421)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.00104, -0.02161, -0.00255, 0.13439, 0.19543)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.04737, -0.18450, 0.10472, -0.70153, -0.66218)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0107, 0.01204, 0.01204, -0.01338),
              doubleArrayOf(-0.01107, -0.01246, -0.01246, 0.01384),
              doubleArrayOf(-0.00377, -0.00424, -0.00424, 0.00471),
              doubleArrayOf(0.20446, 0.23001, 0.23001, -0.25557),
              doubleArrayOf(-0.34737, -0.39079, -0.39079, 0.43421)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.00084, -0.00094, -0.00094, 0.00104),
              doubleArrayOf(0.01729, 0.01945, 0.01945, -0.02161),
              doubleArrayOf(0.00204, 0.00230, 0.00230, -0.00255),
              doubleArrayOf(-0.10751, -0.12095, -0.12095, 0.13439),
              doubleArrayOf(-0.15635, -0.17589, -0.17589, 0.19543)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.03789, 0.04263, 0.04263, -0.04737),
              doubleArrayOf(0.1476, 0.16605, 0.16605, -0.1845),
              doubleArrayOf(-0.08378, -0.09425, -0.09425, 0.10472),
              doubleArrayOf(0.56122, 0.63138, 0.63138, -0.70153),
              doubleArrayOf(0.52975, 0.59596, 0.59596, -0.66218)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.00268, -0.00268, 0.00401, 0.01204, 0.0107),
              doubleArrayOf(-0.00277, 0.00277, -0.00415, -0.01246, -0.01107),
              doubleArrayOf(-0.00094, 0.00094, -0.00141, -0.00424, -0.00377),
              doubleArrayOf(0.05111, -0.05111, 0.07667, 0.23001, 0.20446),
              doubleArrayOf(-0.08684, 0.08684, -0.13026, -0.39079, -0.34737)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.00021, 0.00021, -0.00031, -0.00094, -0.00084),
              doubleArrayOf(0.00432, -0.00432, 0.00648, 0.01945, 0.01729),
              doubleArrayOf(0.00051, -0.00051, 0.00077, 0.00230, 0.00204),
              doubleArrayOf(-0.02688, 0.02688, -0.04032, -0.12095, -0.10751),
              doubleArrayOf(-0.03909, 0.03909, -0.05863, -0.17589, -0.15635)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.27934, 0.57798, -0.43002, 0.26446)),
            tolerance = 1.0e-05))
        }
      }

      on("with next state only") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayerContextWindow.Front())
        val paramsErrors = RANLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = RANLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.33625, -0.42516, 0.59996, -1.5982, -1.20369)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.08207, 0.00801, 0.015, -0.67508, 0.42049)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.13333, -0.10698, 0.29998, -1.12633, -0.54484)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.08207, 0.00801, 0.015, -0.67508, 0.42049)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.13333, -0.10698, 0.29998, -1.12633, -0.54484)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.06566, 0.07386, 0.07386, -0.08207),
              doubleArrayOf(-0.0064, -0.00721, -0.00721, 0.00801),
              doubleArrayOf(-0.012, -0.0135, -0.0135, 0.015),
              doubleArrayOf(0.54007, 0.60757, 0.60757, -0.67508),
              doubleArrayOf(-0.3364, -0.37844, -0.37844, 0.42049)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.10666, 0.12, 0.12, -0.13333),
              doubleArrayOf(0.08558, 0.09628, 0.09628, -0.10698),
              doubleArrayOf(-0.23998, -0.26998, -0.26998, 0.29998),
              doubleArrayOf(0.90106, 1.01369, 1.01369, -1.12633),
              doubleArrayOf(0.43587, 0.49036, 0.49036, -0.54484)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.75531, 0.92351, -0.71463, 0.20447)),
            tolerance = 1.0e-05))
        }
      }

      on("with previous and next state") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayerContextWindow.Bilateral())
        val paramsErrors = RANLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = RANLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2155, -0.38879, 0.59046, -1.70496, -1.42786)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.04401, 0.00728, 0.01461, -0.50966, 0.50086)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.00344, -0.01137, -0.00791, 0.268, 0.22543)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.15584, -0.0971, 0.32465, -1.39899, -0.76382)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.04401, 0.00728, 0.01461, -0.50966, 0.50086)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.00344, -0.01137, -0.00791, 0.268, 0.22543)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.15584, -0.0971, 0.32465, -1.39899, -0.76382)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.03521, 0.03961, 0.03961, -0.04401),
              doubleArrayOf(-0.00583, -0.00656, -0.00656, 0.00728),
              doubleArrayOf(-0.01169, -0.01315, -0.01315, 0.01461),
              doubleArrayOf(0.40773, 0.45869, 0.45869, -0.50966),
              doubleArrayOf(-0.40069, -0.45078, -0.45078, 0.50086)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.00275, -0.00309, -0.00309, 0.00344),
              doubleArrayOf(0.0091, 0.01024, 0.01024, -0.01137),
              doubleArrayOf(0.00633, 0.00712, 0.00712, -0.00791),
              doubleArrayOf(-0.2144, -0.2412, -0.2412, 0.268),
              doubleArrayOf(-0.18034, -0.20289, -0.20289, 0.22543)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.12467, 0.14025, 0.14025, -0.15584),
              doubleArrayOf(0.07768, 0.08739, 0.08739, -0.0971),
              doubleArrayOf(-0.25972, -0.29219, -0.29219, 0.32465),
              doubleArrayOf(1.11919, 1.25909, 1.25909, -1.39899),
              doubleArrayOf(0.61106, 0.68744, 0.68744, -0.76382)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0088, -0.0088, 0.0132, 0.03961, 0.03521),
              doubleArrayOf(-0.00146, 0.00146, -0.00219, -0.00656, -0.00583),
              doubleArrayOf(-0.00292, 0.00292, -0.00438, -0.01315, -0.01169),
              doubleArrayOf(0.10193, -0.10193, 0.1529, 0.45869, 0.40773),
              doubleArrayOf(-0.10017, 0.10017, -0.15026, -0.45078, -0.40069)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.00069, 0.00069, -0.00103, -0.00309, -0.00275),
              doubleArrayOf(0.00227, -0.00227, 0.00341, 0.01024, 0.0091),
              doubleArrayOf(0.00158, -0.00158, 0.00237, 0.00712, 0.00633),
              doubleArrayOf(-0.0536, 0.0536, -0.0804, -0.2412, -0.2144),
              doubleArrayOf(-0.04509, 0.04509, -0.06763, -0.20289, -0.18034)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.71206, 1.22819, -1.06908, -0.06037)),
            tolerance = 1.0e-05))
        }
      }
    }
  }
})
