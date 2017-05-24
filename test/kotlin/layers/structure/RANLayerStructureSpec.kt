/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package layers.structure

import com.kotlinnlp.simplednn.core.layers.recurrent.ran.RANLayerParameters
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.simplemath.NDArray
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
            NDArray.arrayOf(doubleArrayOf(0.397, 0.252, 0.5, 0.705, 0.453)),
            tolerance = 0.0005))
        }

        it("should match the expected forget gate") {
          assertEquals(true, layer.forgetGate.values.equals(
            NDArray.arrayOf(doubleArrayOf(0.853, 0.433, 0.116, 0.52, 0.242)),
            tolerance = 0.0005))
        }

        it("should match the expected candidate") {
          assertEquals(true, layer.candidate.values.equals(
            NDArray.arrayOf(doubleArrayOf(1.02, -0.1, 0.1, 2.03, -1.41)),
            tolerance = 0.0005))
        }

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            NDArray.arrayOf(doubleArrayOf(0.384, -0.025, 0.05, 0.892, -0.564)),
            tolerance = 0.0005))
        }
      }

      on("with previous state context") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayerContextWindow.Back())
        layer.forward()

        it("should match the expected input gate") {
          assertEquals(true, layer.inputGate.values.equals(
            NDArray.arrayOf(doubleArrayOf(0.723, 0.25, 0.55, 0.821, 0.535)),
            tolerance = 0.0005))
        }

        it("should match the expected forget gate") {
          assertEquals(true, layer.forgetGate.values.equals(
            NDArray.arrayOf(doubleArrayOf(0.911, 0.181, 0.048, 0.675, 0.389)),
            tolerance = 0.0005))
        }

        it("should match the expected candidate") {
          assertEquals(true, layer.candidate.values.equals(
            NDArray.arrayOf(doubleArrayOf(1.02, -0.1, 0.1, 2.03, -1.41)),
            tolerance = 0.0005))
        }

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            NDArray.arrayOf(doubleArrayOf(0.504, 0.011, 0.04, 0.785, -0.788)),
            tolerance = 0.0005))
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
            NDArray.arrayOf(doubleArrayOf(-0.186, -0.775, 0.2, -0.748, -1.014)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            NDArray.arrayOf(doubleArrayOf(-0.045, 0.015, 0.005, -0.316, 0.354)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            NDArray.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            NDArray.arrayOf(doubleArrayOf(-0.074, -0.195, 0.1, -0.527, -0.459)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            NDArray.arrayOf(doubleArrayOf(-0.045, 0.015, 0.005, -0.316, 0.354)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            NDArray.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            NDArray.arrayOf(doubleArrayOf(-0.074, -0.195, 0.1, -0.527, -0.459)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, paramsErrors.inputGate.weights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.036, 0.041, 0.041, -0.045),
              doubleArrayOf(-0.012, -0.013, -0.013, 0.015),
              doubleArrayOf(-0.004, -0.004, -0.004, 0.005),
              doubleArrayOf(0.253, 0.284, 0.284, -0.316),
              doubleArrayOf(-0.283, -0.319, -0.319, 0.354)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, paramsErrors.forgetGate.weights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, paramsErrors.candidate.weights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.059, 0.066, 0.066, -0.074),
              doubleArrayOf(0.156, 0.176, 0.176, -0.195),
              doubleArrayOf(-0.08, -0.09, -0.09, 0.1),
              doubleArrayOf(0.422, 0.475, 0.475, -0.527),
              doubleArrayOf(0.367, 0.413, 0.413, -0.459)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            NDArray.arrayOf(doubleArrayOf(0.418, 0.340, -0.212, 0.392)),
            tolerance = 0.0005))
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
            NDArray.arrayOf(doubleArrayOf(-0.066, -0.739, 0.19, -0.855, -1.238)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            NDArray.arrayOf(doubleArrayOf(-0.013, 0.014, 0.005, -0.256, 0.434)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            NDArray.arrayOf(doubleArrayOf(0.001, -0.022, -0.003, 0.134, 0.195)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            NDArray.arrayOf(doubleArrayOf(-0.047, -0.185, 0.105, -0.702, -0.662)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            NDArray.arrayOf(doubleArrayOf(-0.013, 0.014, 0.005, -0.256, 0.434)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            NDArray.arrayOf(doubleArrayOf(0.001, -0.022, -0.003, 0.134, 0.195)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            NDArray.arrayOf(doubleArrayOf(-0.047, -0.185, 0.105, -0.702, -0.662)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, paramsErrors.inputGate.weights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.011, 0.012, 0.012, -0.013),
              doubleArrayOf(-0.011, -0.012, -0.012, 0.014),
              doubleArrayOf(-0.004, -0.004, -0.004, 0.005),
              doubleArrayOf(0.204, 0.23, 0.23, -0.256),
              doubleArrayOf(-0.347, -0.391, -0.391, 0.434)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, paramsErrors.forgetGate.weights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(-0.001, -0.001, -0.001, 0.001),
              doubleArrayOf(0.017, 0.019, 0.019, -0.022),
              doubleArrayOf(0.002, 0.002, 0.002, -0.003),
              doubleArrayOf(-0.108, -0.121, -0.121, 0.134),
              doubleArrayOf(-0.156, -0.176, -0.176, 0.195)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, paramsErrors.candidate.weights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.038, 0.043, 0.043, -0.047),
              doubleArrayOf(0.148, 0.166, 0.166, -0.185),
              doubleArrayOf(-0.084, -0.094, -0.094, 0.105),
              doubleArrayOf(0.561, 0.631, 0.631, -0.702),
              doubleArrayOf(0.530, 0.596, 0.596, -0.662)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.003, -0.003, 0.004, 0.012, 0.011),
              doubleArrayOf(-0.003, 0.003, -0.004, -0.012, -0.011),
              doubleArrayOf(-0.001, 0.001, -0.001, -0.004, -0.004),
              doubleArrayOf(0.051, -0.051, 0.077, 0.23, 0.204),
              doubleArrayOf(-0.087, 0.087, -0.13, -0.391, -0.347)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, -0.001, -0.001),
              doubleArrayOf(0.004, -0.004, 0.006, 0.019, 0.017),
              doubleArrayOf(0.001, -0.001, 0.001, 0.002, 0.002),
              doubleArrayOf(-0.027, 0.027, -0.04, -0.121, -0.108),
              doubleArrayOf(-0.039, 0.039, -0.059, -0.176, -0.156)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            NDArray.arrayOf(doubleArrayOf(0.279, 0.578, -0.430, 0.264)),
            tolerance = 0.0005))
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
            NDArray.arrayOf(doubleArrayOf(-0.336, -0.425, 0.600, -1.598, -1.204)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            NDArray.arrayOf(doubleArrayOf(-0.082, 0.008, 0.015, -0.675, 0.420)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            NDArray.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            NDArray.arrayOf(doubleArrayOf(-0.133, -0.107, 0.300, -1.126, -0.545)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            NDArray.arrayOf(doubleArrayOf(-0.082, 0.008, 0.015, -0.675, 0.420)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            NDArray.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            NDArray.arrayOf(doubleArrayOf(-0.133, -0.107, 0.300, -1.126, -0.545)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, paramsErrors.inputGate.weights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.066, 0.074, 0.074, -0.082),
              doubleArrayOf(-0.006, -0.007, -0.007, 0.008),
              doubleArrayOf(-0.012, -0.013, -0.013, 0.015),
              doubleArrayOf(0.540, 0.608, 0.608, -0.675),
              doubleArrayOf(-0.336, -0.378, -0.378, 0.420)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, paramsErrors.forgetGate.weights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, paramsErrors.candidate.weights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.107, 0.120, 0.120, -0.133),
              doubleArrayOf(0.086, 0.096, 0.096, -0.107),
              doubleArrayOf(-0.240, -0.270, -0.270, 0.300),
              doubleArrayOf(0.901, 1.014, 1.014, -1.126),
              doubleArrayOf(0.436, 0.490, 0.490, -0.545)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            NDArray.arrayOf(doubleArrayOf(0.755, 0.924, -0.715, 0.204)),
            tolerance = 0.0005))
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
            NDArray.arrayOf(doubleArrayOf(-0.216, -0.389, 0.590, -1.705, -1.428)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            NDArray.arrayOf(doubleArrayOf(-0.044, 0.007, 0.015, -0.51, 0.501)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            NDArray.arrayOf(doubleArrayOf(0.003, -0.011, -0.008, 0.268, 0.225)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            NDArray.arrayOf(doubleArrayOf(-0.156, -0.097, 0.325, -1.399, -0.764)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            NDArray.arrayOf(doubleArrayOf(-0.044, 0.007, 0.015, -0.51, 0.501)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            NDArray.arrayOf(doubleArrayOf(0.003, -0.011, -0.008, 0.268, 0.225)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            NDArray.arrayOf(doubleArrayOf(-0.156, -0.097, 0.325, -1.399, -0.764)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, paramsErrors.inputGate.weights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.035, 0.04, 0.04, -0.044),
              doubleArrayOf(-0.006, -0.007, -0.007, 0.007),
              doubleArrayOf(-0.012, -0.013, -0.013, 0.015),
              doubleArrayOf(0.408, 0.459, 0.459, -0.51),
              doubleArrayOf(-0.401, -0.451, -0.451, 0.501)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, paramsErrors.forgetGate.weights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(-0.003, -0.003, -0.003, 0.003),
              doubleArrayOf(0.009, 0.01, 0.01, -0.011),
              doubleArrayOf(0.006, 0.007, 0.007, -0.008),
              doubleArrayOf(-0.214, -0.241, -0.241, 0.268),
              doubleArrayOf(-0.18, -0.203, -0.203, 0.225)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, paramsErrors.candidate.weights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.125, 0.14, 0.14, -0.156),
              doubleArrayOf(0.078, 0.087, 0.087, -0.097),
              doubleArrayOf(-0.260, -0.292, -0.292, 0.325),
              doubleArrayOf(1.119, 1.259, 1.259, -1.399),
              doubleArrayOf(0.611, 0.687, 0.687, -0.764)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.009, -0.009, 0.013, 0.04, 0.035),
              doubleArrayOf(-0.001, 0.001, -0.002, -0.007, -0.006),
              doubleArrayOf(-0.003, 0.003, -0.004, -0.013, -0.012),
              doubleArrayOf(0.102, -0.102, 0.153, 0.459, 0.408),
              doubleArrayOf(-0.1, 0.1, -0.15, -0.451, -0.401)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(-0.001, 0.001, -0.001, -0.003, -0.003),
              doubleArrayOf(0.002, -0.002, 0.003, 0.01, 0.009),
              doubleArrayOf(0.002, -0.002, 0.002, 0.007, 0.006),
              doubleArrayOf(-0.054, 0.054, -0.08, -0.241, -0.214),
              doubleArrayOf(-0.045, 0.045, -0.068, -0.203, -0.18)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            NDArray.arrayOf(doubleArrayOf(0.712, 1.228, -1.069, -0.06)),
            tolerance = 0.0005))
        }
      }
    }
  }
})
