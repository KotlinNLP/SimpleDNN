package com.kotlinnlp.simplednn.core.neuralprocessor.recurrent

import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessorsPool
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * A pool of [NeuralProcessor]s which allows to allocate and release processors when needed, without creating a new one.
 * It is useful to optimize the creation of new structures every time a processor is created.
 *
 * @property neuralNetwork the [NeuralNetwork] which the processors of the pool will work with
 */
class RecurrentNeuralProcessorsPool<InputNDArrayType : NDArray<InputNDArrayType>>(
  val neuralNetwork: NeuralNetwork
) : NeuralProcessorsPool<RecurrentNeuralProcessor<InputNDArrayType>>(neuralNetwork) {

  /**
   * The factory of a new processor
   *
   * @param id the id of the processor to create
   *
   * @return a new [RecurrentNeuralProcessor] with the given [id]
   */
  override fun processorFactory(id: Int) = RecurrentNeuralProcessor<InputNDArrayType>(
    neuralNetwork = this.neuralNetwork,
    id = id
  )
}
