/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.ltm

import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class LTMLayerStructureSpec : Spek({

  describe("a LTMLayer") {

    context("forward") {

      context("without previous state context") {

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

      context("with previous state context") {

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

      context("without previous and next state") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayerContextWindow.Empty())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

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
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.019699, -0.043414, 0.063562, -0.241845)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.178839, 0.073758, 0.229447, -0.078241)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.036513, 0.006013, 0.006659, -0.008466)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.008881, 0.004556, 0.011772, -0.013763)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.010163, -0.037099, 0.010209, -0.083935)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.cell.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.019699, -0.043414, 0.063562, -0.241845)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate1.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.036513, 0.006013, 0.006659, -0.008466)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate2.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.008881, 0.004556, 0.011772, -0.013763)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate3.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.010163, -0.037099, 0.010209, -0.083935)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.02921, -0.032861, -0.032861, 0.036513),
                doubleArrayOf(-0.00481, -0.005411, -0.005411, 0.006013),
                doubleArrayOf(-0.005327, -0.005993, -0.005993, 0.006659),
                doubleArrayOf(0.006773, 0.007619, 0.007619, -0.008466)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.007105, -0.007993, -0.007993, 0.008881),
                doubleArrayOf(-0.003645, -0.004101, -0.004101, 0.004556),
                doubleArrayOf(-0.009418, -0.010595, -0.010595, 0.011772),
                doubleArrayOf(0.01101, 0.012387, 0.012387, -0.013763)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.008131, 0.009147, 0.009147, -0.010163),
                doubleArrayOf(0.029679, 0.033389, 0.033389, -0.037099),
                doubleArrayOf(-0.008167, -0.009188, -0.009188, 0.010209),
                doubleArrayOf(0.067148, 0.075541, 0.075541, -0.083935)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.006664, -0.002146, -0.001143, -0.007219),
                doubleArrayOf(-0.014687, -0.004729, -0.002520, -0.015909),
                doubleArrayOf(0.021504, 0.006924, 0.003689, 0.023293),
                doubleArrayOf(-0.081819, -0.026344, -0.014038, -0.088627)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.168188, 0.038366, -0.101206, -0.069296)),
              tolerance = 1.0e-06)
          }
        }
      }

      context("with previous state only") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayerContextWindow.Back())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

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
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.002315, -0.035905, 0.049241, -0.207956)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.157387, 0.071769, 0.19399, -0.0643621)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.032388, 0.006239, 0.006975, -0.0101)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.011706, 0.005928, 0.008345, -0.008952)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.001532, -0.019882, 0.015325, -0.090501)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.cell.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.002315, -0.035905, 0.049241, -0.207956)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate1.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.032388, 0.006239, 0.006975, -0.0101)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate2.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.011706, 0.005928, 0.008345, -0.008952)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate3.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.001532, -0.019882, 0.015325, -0.090501)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.032388, -0.022671, -0.038865, 0.003239),
                doubleArrayOf(-0.006239, -0.004368, -0.007487, 0.000624),
                doubleArrayOf(-0.006975, -0.004882, -0.00837, 0.000697),
                doubleArrayOf(0.0101, 0.00707, 0.01212, -0.00101)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.011706, -0.008194, -0.014047, 0.001171),
                doubleArrayOf(-0.005928, -0.004149, -0.007113, 0.000593),
                doubleArrayOf(-0.008345, -0.005842, -0.010014, 0.000835),
                doubleArrayOf(0.008952, 0.006266, 0.010742, -0.000895)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.001532, 0.001072, 0.001838, -0.000153),
                doubleArrayOf(0.019882, 0.013917, 0.023858, -0.001988),
                doubleArrayOf(-0.015325, -0.010727, -0.01839, 0.001532),
                doubleArrayOf(0.090501, 0.063351, 0.108601, -0.00905)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.00277, -0.00113, -0.001812, -0.002254),
                doubleArrayOf(-0.042967, -0.017523, -0.028108, -0.034958),
                doubleArrayOf(0.058926, 0.024032, 0.038549, 0.047942),
                doubleArrayOf(-0.248855, -0.101492, -0.162798, -0.202468)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.14713, 0.054143, -0.111281, -0.071078)),
              tolerance = 1.0e-06)
          }
        }
      }

      context("with next state only") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayerContextWindow.Front())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

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
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.113795, 0.103172, 0.068124, -0.302068)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.267918, 0.127946, 0.15616, -0.158076)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0547, 0.01043, 0.004532, -0.017104)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.013305, 0.007903, 0.008012, -0.027806)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.05871, -0.053875, 0.004795, -0.062211)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.cell.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.113795, 0.103172, 0.068124, -0.302068)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate1.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0547, 0.01043, 0.004532, -0.017104)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate2.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.013305, 0.007903, 0.008012, -0.027806)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate3.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.05871, -0.053875, 0.004795, -0.062211)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.04376, -0.04923, -0.04923, 0.0547),
                doubleArrayOf(-0.008344, -0.009387, -0.009387, 0.01043),
                doubleArrayOf(-0.003626, -0.004079, -0.004079, 0.004532),
                doubleArrayOf(0.013683, 0.015393, 0.015393, -0.017104)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.010644, -0.011975, -0.011975, 0.013305),
                doubleArrayOf(-0.006323, -0.007113, -0.007113, 0.007903),
                doubleArrayOf(-0.00641, -0.007211, -0.007211, 0.008012),
                doubleArrayOf(0.022245, 0.025026, 0.025026, -0.027806)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.046968, -0.052839, -0.052839, 0.05871),
                doubleArrayOf(0.0431, 0.048488, 0.048488, -0.053875),
                doubleArrayOf(-0.003836, -0.004315, -0.004315, 0.004795),
                doubleArrayOf(0.049769, 0.055990, 0.055990, -0.062211)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.038498, 0.012395, 0.006605, 0.041701),
                doubleArrayOf(0.034904, 0.011238, 0.005989, 0.037808),
                doubleArrayOf(0.023047, 0.007421, 0.003954, 0.024965),
                doubleArrayOf(-0.102193, -0.032903, -0.017533, -0.110696)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.10429, 0.034473, -0.089364, -0.044435)),
              tolerance = 1.0e-06)
          }
        }
      }

      context("with previous and next state") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayerContextWindow.Bilateral())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

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
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.115967, 0.040169, 0.058, -0.268914)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.236542, 0.138204, 0.179305, -0.112362)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.048676, 0.012015, 0.006447, -0.017632)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.017594, 0.011415, 0.007713, -0.015628)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.076743, -0.028509, 0.006365, -0.071562)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.cell.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.115967, 0.040169, 0.058, -0.268914)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate1.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.048676, 0.012015, 0.006447, -0.017632)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate2.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.017594, 0.011415, 0.007713, -0.015628)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate3.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.076743, -0.028509, 0.006365, -0.071562)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.048676, -0.034074, -0.058412, 0.004868),
                doubleArrayOf(-0.012015, -0.008411, -0.014418, 0.001202),
                doubleArrayOf(-0.006447, -0.004513, -0.007736, 0.000645),
                doubleArrayOf(0.017632, 0.012343, 0.021159, -0.001763)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.017594, -0.012316, -0.021112, 0.001759),
                doubleArrayOf(-0.011415, -0.007991, -0.013698, 0.001142),
                doubleArrayOf(-0.007713, -0.005399, -0.009256, 0.000771),
                doubleArrayOf(0.015628, 0.010940, 0.018754, -0.001563)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.076743, -0.053720, -0.092092, 0.007674),
                doubleArrayOf(0.028509, 0.019957, 0.034211, -0.002851),
                doubleArrayOf(-0.006365, -0.004455, -0.007638, 0.000636),
                doubleArrayOf(0.071562, 0.050093, 0.085874, -0.007156)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.138775, 0.056598, 0.090785, 0.112907),
                doubleArrayOf(0.048069, 0.019604, 0.031446, 0.039109),
                doubleArrayOf(0.069407, 0.028307, 0.045405, 0.056470),
                doubleArrayOf(-0.321803, -0.131243, -0.210520, -0.261819)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.069055, 0.067126, -0.107279, -0.053312)),
              tolerance = 1.0e-06)
          }
        }
      }
    }
  }
})
