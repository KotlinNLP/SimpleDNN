package utils.exampleextractor

import com.jsoniter.JsonIterator
import com.jsoniter.ValueType
import com.kotlinnlp.simplednn.dataset.SimpleExample
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import utils.readSparseBinaryNDArray

/**
 *
 */
class ClassificationSparseExampleExtractor(
  val inputSize: Int,
  val outputSize: Int
) : ExampleExtractor<SimpleExample<SparseBinaryNDArray>> {

  /**
   *
   */
  override fun extract(iterator: JsonIterator): SimpleExample<SparseBinaryNDArray> {

    val outputGold = DenseNDArrayFactory.zeros(Shape(this.outputSize))
    var goldIndex: Int
    var features: SparseBinaryNDArray? = null

    while (iterator.readArray()) {

      if (iterator.whatIsNext() == ValueType.ARRAY) {
        features = iterator.readSparseBinaryNDArray(size = inputSize)

      } else if (iterator.whatIsNext() == ValueType.NUMBER) {
        goldIndex = iterator.readInt()
        outputGold[goldIndex] = 1.0
      }
    }

    return SimpleExample(features!!, outputGold)
  }
}
