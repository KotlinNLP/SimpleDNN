/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package layers.structure

import com.kotlinnlp.simplednn.core.layers.recurrent.deltarnn.DeltaRNNLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import layers.structure.utils.DeltaRNNLayerStructureUtils
import layers.structure.contextwindows.DeltaLayerContextWindow
import kotlin.test.assertTrue

/**
 *
 */
class DeltaRNNLayerStructureSpec : Spek({

  describe("a DeltaRNNLayerStructure") {

    context("forward") {

      on("without previous state context") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayerContextWindow.Empty())
        layer.forward()

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.568971, 0.410323, -0.39693, 0.648091, -0.449441)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected partition array") {
          assertTrue {
            layer.partition.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.519989, 0.169384, 0.668188, 0.325195, 0.601088)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected output") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.287518, 0.06939, -0.259175, 0.20769, -0.263768)),
              tolerance = 1.0e-06)
          }
        }
      }

      on("with previous state context") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayerContextWindow.Back())
        layer.forward()

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.576403, 0.40594, -0.222741, 0.36182, -0.42253)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected partition array") {
          assertTrue {
            layer.partition.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.519989, 0.169384, 0.668188, 0.325195, 0.601088)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected output") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.202158, 0.228591, -0.240679, -0.350224, -0.476828)),
              tolerance = 1.0e-06)
          }
        }
      }
    }

    context("backward") {

      on("without previous and next state") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayerContextWindow.Empty())
        val paramsErrors = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(DeltaRNNLayerStructureUtils.getOutputGold()))
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.282482, -0.68061, -0.109175, -1.43231, -0.713768)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate array") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.099336, -0.095874, -0.061456, -0.270142, -0.342373)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition array") {
          assertTrue {
            layer.partition.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.040117, -0.039291, 0.009608, -0.203702, 0.076921)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.feedforwardUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.099336, -0.095874, -0.061456, -0.270142, -0.342373)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition biases") {
          assertTrue {
            paramsErrors.recurrentUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.040117, -0.039291, 0.009608, -0.203702, 0.076921)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the alpha array") {
          assertTrue {
            paramsErrors.alpha.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta1 array") {
          assertTrue {
            paramsErrors.beta1.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.081455, 0.104503, -0.018437, -0.01891, -0.071898)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta2 array") {
          assertTrue {
            paramsErrors.beta2.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.feedforwardUnit.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(0.008253, 0.009284, 0.009284, -0.010316),
                doubleArrayOf(0.000753, 0.000847, 0.000847, -0.000942),
                doubleArrayOf(-0.027352, -0.030771, -0.030771, 0.03419),
                doubleArrayOf(0.076516, 0.086081, 0.086081, -0.095645),
                doubleArrayOf(-0.171096, -0.192483, -0.192483, 0.21387)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the recurrent weights") {
          val wRec: DenseNDArray = paramsErrors.recurrentUnit.weights.values as DenseNDArray
          assertTrue {
            wRec.equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.027148, 0.270204, -0.131293, 0.204699)),
              tolerance = 1.0e-06)
          }
        }
      }

      on("with previous state only") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayerContextWindow.Back())
        val paramsErrors = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(DeltaRNNLayerStructureUtils.getOutputGold()))
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.367842, -0.521409, -0.090679, -1.990224, -0.926828)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate array") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.127725, -0.073765, -0.057584, -0.562482, -0.457644)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition array") {
          assertTrue {
            layer.partition.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.071043, -0.0153, -0.001379, -0.470858, -0.053671)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.feedforwardUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.127725, -0.073765, -0.057584, -0.562482, -0.457644)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition biases") {
          assertTrue {
            paramsErrors.recurrentUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.071043, -0.0153, -0.001379, -0.470858, -0.053671)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the alpha array") {
          assertTrue {
            paramsErrors.alpha.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.115834, -0.00333, -0.003066, -0.020045, -0.026394)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta1 array") {
          assertTrue {
            paramsErrors.beta1.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.104734, 0.080403, -0.017275, -0.039374, -0.096105)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta2 array") {
          assertTrue {
            paramsErrors.beta2.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.14126, 0.003055, -0.01022, -0.286357, -0.125688)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.feedforwardUnit.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(-0.030323, -0.034114, -0.034114, 0.037904),
                doubleArrayOf(-0.010631, -0.01196, -0.01196, 0.013289),
                doubleArrayOf(-0.014871, -0.01673, -0.01673, 0.018589),
                doubleArrayOf(0.288327, 0.324368, 0.324368, -0.360408),
                doubleArrayOf(-0.093454, -0.105136, -0.105136, 0.116817)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the recurrent weights") {
          val wRec: DenseNDArray = paramsErrors.recurrentUnit.weights.values as DenseNDArray
          assertTrue {
            wRec.equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(0.000252, -0.000252, 0.000372, 0.000915, 0.000848),
                doubleArrayOf(0.001849, -0.001849, 0.002729, 0.00671, 0.006221),
                doubleArrayOf(0.012389, -0.012389, 0.018285, 0.04496, 0.041679),
                doubleArrayOf(-0.085707, 0.085707, -0.126498, -0.311042, -0.288349),
                doubleArrayOf(0.01093, -0.01093, 0.016131, 0.039665, 0.036771)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.200333, 0.4456, -0.10519, 0.105415)),
              tolerance = 1.0e-06)
          }
        }
      }

      on("with next state only") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayerContextWindow.Front())
        val paramsErrors = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(DeltaRNNLayerStructureUtils.getOutputGold()))
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.082082, -0.40561, -0.205775, -0.40071, -0.751368)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate array") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.028864, -0.057136, -0.115833, -0.075576, -0.360408)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition array") {
          assertTrue {
            layer.partition.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.011657, -0.023416, 0.018109, -0.056989, 0.080973)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.feedforwardUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.028864, -0.057136, -0.115833, -0.075576, -0.360408)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition biases") {
          assertTrue {
            paramsErrors.recurrentUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.011657, -0.023416, 0.018109, -0.056989, 0.080973)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the alpha array") {
          assertTrue {
            paramsErrors.alpha.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta1 array") {
          assertTrue {
            paramsErrors.beta1.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.023669, 0.062279, -0.03475, -0.00529, -0.075686)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta2 array") {
          assertTrue {
            paramsErrors.beta2.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.feedforwardUnit.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(0.002398, 0.002698, 0.002698, -0.002998),
                doubleArrayOf(0.000449, 0.000505, 0.000505, -0.000561),
                doubleArrayOf(-0.051554, -0.057998, -0.057998, 0.064443),
                doubleArrayOf(0.021407, 0.024082, 0.024082, -0.026758),
                doubleArrayOf(-0.180109, -0.202623, -0.202623, 0.225136)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the recurrent weights") {
          val wRec: DenseNDArray = paramsErrors.recurrentUnit.weights.values as DenseNDArray
          assertTrue {
            wRec.equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.111866, 0.202535, -0.135921, 0.217254)),
              tolerance = 1.0e-06)
          }
        }
      }

      on("with previous and next state") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayerContextWindow.Bilateral())
        val paramsErrors = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(DeltaRNNLayerStructureUtils.getOutputGold()))
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.167442, -0.246409, -0.187279, -0.958624, -0.964428)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate array") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.058141, -0.03486, -0.118929, -0.270929, -0.47621)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition array") {
          assertTrue {
            layer.partition.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.032339, -0.00723, -0.002847, -0.226797, -0.055849)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.feedforwardUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.058141, -0.03486, -0.118929, -0.270929, -0.47621)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition biases") {
          assertTrue {
            paramsErrors.recurrentUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.032339, -0.00723, -0.002847, -0.226797, -0.055849)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the alpha array") {
          assertTrue {
            paramsErrors.alpha.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.052728, -0.001574, -0.006332, -0.009655, -0.027465)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta1 array") {
          assertTrue {
            paramsErrors.beta1.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.047675, 0.037997, -0.035679, -0.018965, -0.100004)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta2 array") {
          assertTrue {
            paramsErrors.beta2.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.064302, 0.001444, -0.021108, -0.137928, -0.130787)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.feedforwardUnit.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(-0.013803, -0.015529, -0.015529, 0.017254),
                doubleArrayOf(-0.005024, -0.005652, -0.005652, 0.006280),
                doubleArrayOf(-0.030713, -0.034553, -0.034553, 0.038392),
                doubleArrayOf(0.138877, 0.156237, 0.156237, -0.173597),
                doubleArrayOf(-0.097245, -0.109401, -0.109401, 0.121556)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the recurrent weights") {
          val wRec: DenseNDArray = paramsErrors.recurrentUnit.weights.values as DenseNDArray
          assertTrue {
            wRec.equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(0.000115, -0.000115, 0.000169, 0.000416, 0.000386),
                doubleArrayOf(0.000874, -0.000874, 0.00129, 0.003171, 0.00294),
                doubleArrayOf(0.025586, -0.025586, 0.037764, 0.092855, 0.086081),
                doubleArrayOf(-0.041282, 0.041282, -0.06093, -0.149819, -0.138888),
                doubleArrayOf(0.011373, -0.011373, 0.016786, 0.041274, 0.038263)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.050357, 0.25876, -0.086747, 0.118424)),
              tolerance = 1.0e-06)
          }
        }
      }
    }
  }
})
