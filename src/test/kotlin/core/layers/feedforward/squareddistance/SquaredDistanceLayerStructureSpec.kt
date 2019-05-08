package core.layers.feedforward.squareddistance

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

class SquaredDistanceLayerStructureSpec : Spek({

  describe("a Square Distance Layer")
  {

    on("forward") {

      val layer = SquaredDistanceLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.06081)),
              tolerance = 1.0e-05)
        }
      }
    }

    on("backward") {

      val layer = SquaredDistanceLayerUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(SquaredDistanceLayerUtils.getOutputErrors())
      layer.backward(propagateToInput = true)

      it("should match the expected errors of the inputArray1") {
        assertTrue {
          layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.04864, -0.04864, 0.0, 0.04864)),
              tolerance = 1.0e-05)
        }
      }
    }
  }
})