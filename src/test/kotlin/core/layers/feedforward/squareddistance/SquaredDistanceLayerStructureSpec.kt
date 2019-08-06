package core.layers.feedforward.squareddistance

import com.kotlinnlp.simplednn.core.layers.models.feedforward.squareddistance.SquaredDistanceLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

class SquaredDistanceLayerStructureSpec : Spek({

  describe("a Square Distance Layer")
  {

    context("forward") {

      val layer = SquaredDistanceLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5928)),
            tolerance = 1.0e-05)
        }
      }
    }

    context("backward") {

      val layer = SquaredDistanceLayerUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(SquaredDistanceLayerUtils.getOutputErrors())
      val paramsErrors = layer.backward(propagateToInput = true)

      val params = layer.params

      it("should match the expected errors of the inputArray") {
        assertTrue {
          layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.9568, -0.848, 0.5936)),
            tolerance = 1.0e-05)
        }
      }

      it("should match the expected errors of the weights") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.wB)!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.2976, -0.496, 0.3968),
              doubleArrayOf(0.0144, 0.024, -0.0192),
              doubleArrayOf(-0.1488, -0.248, 0.1984),
              doubleArrayOf(-0.1584, -0.264, 0.2112),
              doubleArrayOf(0.024, 0.04, -0.032)
            )),
            tolerance = 1.0e-05)
        }
      }
    }
  }
})
