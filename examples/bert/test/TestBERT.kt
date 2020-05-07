/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package bert.test

import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.transformers.BERT
import com.kotlinnlp.simplednn.deeplearning.transformers.BERTModel
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.WordPieceTokenizer
import java.io.FileInputStream

/**
 * Test a BERT model reconstructing the masked tokens in a text.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) {

  val parsedArgs = CommandLineArguments(args)

  val reconstructor = TextReconstructor(model = parsedArgs.bertModelPath.let {
    println("Reading BERT model from '$it'...")
    BERTModel.load(FileInputStream(it))
  })

  while (true) {
    readInput()?.let {
      println("Reconstructed text: ${reconstructor.reconstruct(it)}")
    } ?: break
  }
}

/**
 * Reconstruct a text with masked words.
 *
 * @param model a BERT transformer model
 */
private class TextReconstructor(model: BERTModel) {

  companion object {

    /**
     * The keyword that indicates the words to reconstruct in the input text.
     */
    const val RECONSTRUCT_KEY = "XXX"
  }

  /**
   * The text tokenizer.
   */
  private val tokenizer = WordPieceTokenizer(model.vocabulary)

  /**
   * A BERT transformer.
   */
  private val bert = BERT(model, masksEnabled = true)

  /**
   * The terms classifier based on the BERT model dictionary.
   */
  private val classifier: FeedforwardNeuralProcessor<DenseNDArray> =
    FeedforwardNeuralProcessor(model = model.classifier, propagateToInput = false, useDropout = false)

  /**
   * Reconstruct the masked tokens of a text.
   *
   * @param text the input text
   *
   * @return the given text with the [RECONSTRUCT_KEY]s replaced with the predictions of the [classifier]
   */
  fun reconstruct(text: String): String {

    val tokens: List<String> = this.tokenizer.tokenize(text)
    val maskedTokens: List<String> = tokens.map { if (it == RECONSTRUCT_KEY) BERTModel.FuncToken.MASK.form else it }

    val encodings: List<DenseNDArray> = this.bert.forward(maskedTokens)

    return tokens.zip(encodings).joinToString(" ") { (token, encoding) ->
      if (token == RECONSTRUCT_KEY) this.reconstructForm(encoding) else token
    }
  }

  /**
   * @param encoding the encoding of a token
   *
   * @return the token form reconstructed with the classifier
   */
  private fun reconstructForm(encoding: DenseNDArray): String {

    val classification: DenseNDArray = this.classifier.forward(encoding)

    return this.bert.model.vocabulary.getElement(id = classification.argMaxIndex())!!
  }
}

/**
 * Read a text from the standard input.
 *
 * @return the string read or null if it was empty
 */
private fun readInput(): String? {

  print("\nInput text (empty to exit): ")

  return readLine()!!.trim().ifEmpty { null }
}
