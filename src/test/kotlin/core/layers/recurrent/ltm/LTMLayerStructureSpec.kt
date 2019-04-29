/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.ltm

import com.kotlinnlp.simplednn.core.layers.models.recurrent.ltm.LTMLayerParameters
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */
class LTMLayerStructureSpec : Spek({

  describe("a LTMLayer") {

    context("forward") {

      on("without previous state context") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayerContextWindow.Empty())
        layer.forward()

        it("should match the expected input gate L1") {
          assertTrue {
            layer.inputGate1.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.396517, 0.251618, 0.5, 0.704746)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected input gate L2") {
          assertTrue {
            layer.inputGate2.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.853210, 0.432907, 0.116089, 0.519989)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected input gate L3") {
          assertTrue {
            layer.inputGate3.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.789182, 0.354344, 0.880797, 0.849412)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected cell") {
          assertTrue {
            layer.cell.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.591378, 0.244436, 0.257845, 0.566106)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.466705, 0.086614, 0.227109, 0.480857)),
              tolerance = 1.0e-06)
          }
        }
      }

      on("with previous state context") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayerContextWindow.Back())
        layer.forward()

        it("should match the expected input gate L1") {
          assertTrue {
            layer.inputGate1.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.593873, 0.349781, 0.305764, 0.650219)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected input gate L2") {
          assertTrue {
            layer.inputGate2.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.853210, 0.382252, 0.169384, 0.689974)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected input gate L3") {
          assertTrue {
            layer.inputGate3.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.798991, 0.509999, 0.766741, 0.694236)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected cell") {
          assertTrue {
            layer.cell.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.696254, 0.115084, 0.250495, 0.297410)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.556301, 0.058693, 0.192065, 0.206473)),
              tolerance = 1.0e-06)
          }
        }
      }
    }

    context("backward") {

      on("without previous and next state") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayerContextWindow.Empty())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params as LTMLayerParameters

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.103295, -0.663386, 0.377109, -1.159143)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.024961, -0.122519, 0.072164, -0.28472)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.203316, 0.107195, 0.326185, -0.055519)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.048652, 0.020185, 0.081546, -0.011552)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.025464, 0.026316, 0.033471, -0.013857)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.017186, -0.151772, 0.039594, -0.148267)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.cell.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.024961, -0.122519, 0.072164, -0.28472)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate1.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.048652, 0.020185, 0.081546, -0.011552)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate2.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.025464, 0.026316, 0.033471, -0.013857)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate3.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.017186, -0.151772, 0.039594, -0.148267)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.038921, -0.043787, -0.043787, 0.048652),
                doubleArrayOf(-0.016148, -0.018167, -0.018167, 0.020185),
                doubleArrayOf(-0.065237, -0.073392, -0.073392, 0.081546),
                doubleArrayOf(0.009242, 0.010397, 0.010397, -0.011552)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.020371, -0.022917, -0.022917, 0.025464),
                doubleArrayOf(-0.021053, -0.023685, -0.023685, 0.026316),
                doubleArrayOf(-0.026776, -0.030124, -0.030124, 0.033471),
                doubleArrayOf(0.011086, 0.012472, 0.012472, -0.013857)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.013749, 0.015467, 0.015467, -0.017186),
                doubleArrayOf(0.121418, 0.136595, 0.136595, -0.151772),
                doubleArrayOf(-0.031675, -0.035635, -0.035635, 0.039594),
                doubleArrayOf(0.118614, 0.13344, 0.13344, -0.148267)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.008445, -0.002719, -0.001449, -0.009147),
                doubleArrayOf(-0.04145, -0.013346, -0.007112, -0.044898),
                doubleArrayOf(0.024414, 0.007861, 0.004189, 0.026445),
                doubleArrayOf(-0.096324, -0.031014, -0.016526, -0.104339)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.426737, -0.050815, -0.135758, -0.047913)),
              tolerance = 1.0e-06)
          }
        }
      }

      on("with previous state only") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayerContextWindow.Back())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params as LTMLayerParameters

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.013699, -0.691307, 0.342065, -1.433527)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.002897, -0.070402, 0.064222, -0.299546)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.225595, 0.109246, 0.291693, -0.080641)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.054411, 0.024846, 0.061918, -0.018341)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.028254, 0.025797, 0.041039, -0.01725)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.0022, -0.172758, 0.061178, -0.304298)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.cell.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.002897, -0.070402, 0.064222, -0.299546)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate1.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.054411, 0.024846, 0.061918, -0.018341)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate2.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.028254, 0.025797, 0.041039, -0.01725)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate3.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.0022, -0.172758, 0.061178, -0.304298)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.054411, -0.038088, -0.065293, 0.005441),
                doubleArrayOf(-0.024846, -0.017392, -0.029816, 0.002485),
                doubleArrayOf(-0.061918, -0.043343, -0.074302, 0.006192),
                doubleArrayOf(0.018341, 0.012838, 0.022009, -0.001834)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.028254, -0.019778, -0.033905, 0.002825),
                doubleArrayOf(-0.025797, -0.018058, -0.030956, 0.00258),
                doubleArrayOf(-0.041039, -0.028727, -0.049247, 0.004104),
                doubleArrayOf(0.01725, 0.012075, 0.0207, -0.001725)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.0022, 0.00154, 0.00264, -0.00022),
                doubleArrayOf(0.172758, 0.12093, 0.207309, -0.017276),
                doubleArrayOf(-0.061178, -0.042825, -0.073414, 0.006118),
                doubleArrayOf(0.304298, 0.213009, 0.365158, -0.03043)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.003467, -0.001414, -0.002268, -0.002821),
                doubleArrayOf(-0.084249, -0.03436, -0.055114, -0.068545),
                doubleArrayOf(0.076852, 0.031343, 0.050276, 0.062527),
                doubleArrayOf(-0.358458, -0.146193, -0.234499, -0.291642)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.584186, 0.043822, -0.285528, -0.164792)),
              tolerance = 1.0e-06)
          }
        }
      }

      on("with next state only") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayerContextWindow.Front())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params as LTMLayerParameters

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.596705, -0.963386, 0.177109, -0.859143)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.144194, -0.011707, 0.072164, -0.333846)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.287529, 0.190182, 0.271923, -0.108747)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.068803, 0.035813, 0.067981, -0.022628)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.036011, 0.046689, 0.027903, -0.027143)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.099276, -0.220407, 0.018595, -0.109894)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.cell.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.144194, -0.011707, 0.072164, -0.333846)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate1.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.068803, 0.035813, 0.067981, -0.022628)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate2.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.036011, 0.046689, 0.027903, -0.027143)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate3.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.099276, -0.220407, 0.018595, -0.109894)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.055042, -0.061923, -0.061923, 0.068803),
                doubleArrayOf(-0.02865, -0.032231, -0.032231, 0.035813),
                doubleArrayOf(-0.054385, -0.061183, -0.061183, 0.067981),
                doubleArrayOf(0.018102, 0.020365, 0.020365, -0.022628)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.028809, -0.03241, -0.03241, 0.036011),
                doubleArrayOf(-0.037352, -0.04202, -0.04202, 0.046689),
                doubleArrayOf(-0.022322, -0.025112, -0.025112, 0.027903),
                doubleArrayOf(0.021715, 0.024429, 0.024429, -0.027143)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.079421, -0.089348, -0.089348, 0.099276),
                doubleArrayOf(0.176326, 0.198367, 0.198367, -0.220407),
                doubleArrayOf(-0.014876, -0.016736, -0.016736, 0.018595),
                doubleArrayOf(0.087915, 0.098904, 0.098904, -0.109894)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.048782, 0.015707, 0.008370, 0.052841),
                doubleArrayOf(-0.00396, -0.001275, -0.000679, -0.004290),
                doubleArrayOf(0.024414, 0.007861, 0.004189, 0.026445),
                doubleArrayOf(-0.112944, -0.036365, -0.019378, -0.122341)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.339681, -0.076956, -0.101, -0.008423)),
              tolerance = 1.0e-06)
          }
        }
      }

      on("with previous and next state") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayerContextWindow.Bilateral())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params as LTMLayerParameters

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.686301, -0.991307, 0.142065, -1.133527)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.145142, -0.009299, 0.064222, -0.341337)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.294746, 0.192276, 0.272064, -0.108926)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.071089, 0.043730, 0.057752, -0.024774)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.036915, 0.045403, 0.038277, -0.023300)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.110223, -0.247728, 0.025408, -0.240616)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.cell.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.145142, -0.009299, 0.064222, -0.341337)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate1.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.071089, 0.043730, 0.057752, -0.024774)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate2.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.036915, 0.045403, 0.038277, -0.023300)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate3.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.110223, -0.247728, 0.025408, -0.240616)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.071089, -0.049762, -0.085307, 0.007109),
                doubleArrayOf(-0.043730, -0.030611, -0.052476, 0.004373),
                doubleArrayOf(-0.057752, -0.040426, -0.069302, 0.005775),
                doubleArrayOf(0.024774, 0.017342, 0.029728, -0.002477)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.036915, -0.025840, -0.044298, 0.003691),
                doubleArrayOf(-0.045403, -0.031782, -0.054484, 0.004540),
                doubleArrayOf(-0.038277, -0.026794, -0.045933, 0.003828),
                doubleArrayOf(0.023300, 0.016310, 0.027960, -0.002330)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.110223, -0.077156, -0.132268, 0.011022),
                doubleArrayOf(0.247728, 0.173409, 0.297273, -0.024773),
                doubleArrayOf(-0.025408, -0.017786, -0.030490, 0.002541),
                doubleArrayOf(0.240616, 0.168431, 0.288740, -0.024062)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.173687, 0.070836, 0.113624, 0.141312),
                doubleArrayOf(-0.011128, -0.004538, -0.007279, -0.009053),
                doubleArrayOf(0.076852, 0.031343, 0.050276, 0.062527),
                doubleArrayOf(-0.408469, -0.166589, -0.267216, -0.332330)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.481427, 0.000129, -0.221932, -0.114356)),
              tolerance = 1.0e-06)
          }
        }
      }
    }
  }
})
