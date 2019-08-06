/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.indrnn

import com.kotlinnlp.simplednn.core.layers.models.recurrent.indrnn.IndRNNLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class IndRNNLayerStructureSpec : Spek({

  describe("an IndRNNLayer") {

    context("forward") {

      context("without previous state context") {

        val layer = IndRNNLayerStructureUtils.buildLayer(IndRNNLayerContextWindow.Empty())
        layer.forward()

        it("should match the expected output") {
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.39693, -0.796878, 0.0, 0.701374, -0.187746)),
            tolerance = 1.0e-06))
        }
      }

      context("with previous state context") {

        val layer = IndRNNLayerStructureUtils.buildLayer(IndRNNLayerContextWindow.Back())

        layer.forward()

        it("should match the expected output") {
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.39693, -0.842046, 0.256335, 0.701374, 0.205456)),
            tolerance = 1.0e-06))
        }
      }
    }

    context("backward") {

      context("without next and previous state") {

        val layer = IndRNNLayerStructureUtils.buildLayer(IndRNNLayerContextWindow.Empty())

        layer.forward()

        layer.outputArray.assignErrors(IndRNNLayerStructureUtils.getOutputErrors())
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.480194, 0.273739, -0.150000, 0.833242, 0.434138)),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.getErrorsOf(params.feedforwardUnit.biases)!!.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.480194, 0.273739, -0.150000, 0.833242, 0.434138)),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.getErrorsOf(params.feedforwardUnit.weights)!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.384155, -0.432175, -0.432175, 0.480194),
              doubleArrayOf(-0.218991, -0.246365, -0.246365, 0.273739),
              doubleArrayOf(0.120000, 0.135000, 0.135000, -0.150000),
              doubleArrayOf(-0.666594, -0.749918, -0.749918, 0.833242),
              doubleArrayOf(-0.347310, -0.390724, -0.390724, 0.434138)
            )),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.getErrorsOf(params.recurrentWeights)!!.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(1.166963, -0.032159, -0.705678, -0.318121)),
            tolerance = 1.0e-06))
        }
      }

      context("with prev state only") {

        val layer = IndRNNLayerStructureUtils.buildLayer(IndRNNLayerContextWindow.Back())

        layer.forward()

        layer.outputArray.assignErrors(IndRNNLayerStructureUtils.getOutputErrors())
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.480194, 0.218219, -0.140144, 0.833242, 0.431005)),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.getErrorsOf(params.feedforwardUnit.biases)!!.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.480194, 0.218219, -0.140144, 0.833242, 0.431005)),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.getErrorsOf(params.feedforwardUnit.weights)!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.384155, -0.432175, -0.432175, 0.480194),
              doubleArrayOf(-0.174576, -0.196397, -0.196397, 0.218219),
              doubleArrayOf(0.112115, 0.126129, 0.126129, -0.140144),
              doubleArrayOf(-0.666594, -0.749918, -0.749918, 0.833242),
              doubleArrayOf(-0.344804, -0.387904, -0.387904, 0.431005)
            )),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.getErrorsOf(params.recurrentWeights)!!.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.094779, 0.043071, 0.040826, -0.596849, -0.286203)),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(1.133745, -0.019984, -0.706080, -0.271285)),
            tolerance = 1.0e-06))
        }
      }

      context("with next state only") {

        val layer = IndRNNLayerStructureUtils.buildLayer(IndRNNLayerContextWindow.Front())

        layer.forward()

        layer.outputArray.assignErrors(IndRNNLayerStructureUtils.getOutputErrors())
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.480194, 0.248190, 0.300000, 0.833242, 0.318368)),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.getErrorsOf(params.feedforwardUnit.biases)!!.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.480194, 0.248190, 0.300000, 0.833242, 0.318368)),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.getErrorsOf(params.feedforwardUnit.weights)!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.384155, -0.432175, -0.432175, 0.480194),
              doubleArrayOf(-0.198552, -0.223371, -0.223371, 0.248190),
              doubleArrayOf(-0.240000, -0.270000, -0.270000, 0.300000),
              doubleArrayOf(-0.666594, -0.749918, -0.749918, 0.833242),
              doubleArrayOf(-0.254694, -0.286531, -0.286531, 0.318368)
            )),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.getErrorsOf(params.recurrentWeights)!!.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(1.417771, -0.452709, -0.492194, -0.165298)),
            tolerance = 1.0e-06))
        }
      }

      context("with next and previous state") {

        val layer = IndRNNLayerStructureUtils.buildLayer(IndRNNLayerContextWindow.Bilateral())

        layer.forward()

        layer.outputArray.assignErrors(IndRNNLayerStructureUtils.getOutputErrors())
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.480194, 0.197852, 0.280288, 0.833242, 0.31607)),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.getErrorsOf(params.feedforwardUnit.biases)!!.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.480194, 0.197852, 0.280288, 0.833242, 0.31607)),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.getErrorsOf(params.feedforwardUnit.weights)!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.384155, -0.432175, -0.432175, 0.480194),
              doubleArrayOf(-0.158282, -0.178067, -0.178067, 0.197852),
              doubleArrayOf(-0.224230, -0.252259, -0.252259, 0.280288),
              doubleArrayOf(-0.666594, -0.749918, -0.749918, 0.833242),
              doubleArrayOf(-0.252856, -0.284463, -0.284463, 0.316070)
            )),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.getErrorsOf(params.recurrentWeights)!!.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.094779, 0.039051, -0.081651, -0.596849, -0.209882)),
            tolerance = 1.0e-06))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(1.367817, -0.421073, -0.501533, -0.136723)),
            tolerance = 1.0e-06))
        }
      }
    }
  }
})
