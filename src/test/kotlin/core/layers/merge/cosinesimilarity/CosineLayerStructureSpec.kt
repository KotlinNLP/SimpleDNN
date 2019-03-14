package core.layers.merge.cosinesimilarity

import com.kotlinnlp.simplednn.core.layers.models.merge.cosinesimilarity.CosineLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import core.layers.merge.distance.DistanceLayerUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */
class CosineLayerStructureSpec: Spek({

  describe("a CosineLayer")
  {

    on("forward") {

      val layer = CosineLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.085901)),
              tolerance = 1.0e-05)
        }
      }
    }

    on("backward") {

      val layer = DistanceLayerUtils.buildLayer()
      val paramsErrors = CosineLayerParameters(inputSize = 4)

      layer.forward()

      layer.outputArray.assignErrors(CosineLayerUtils.getOutputErrors())
      layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

      it("should match the expected errors of the inputArray1") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.04864, -0.04864, 0.0, 0.04864)),
              tolerance = 1.0e-05)
        }
      }

      it("should match the expected errors of the inputArray2") {
        assertTrue {
          layer.inputArrays[1].errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.04864, 0.04864, 0.0, -0.04864)),
              tolerance = 1.0e-05)
        }
      }
    }
  }
})