/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package bert.test

import bert.readDictionary
import com.kotlinnlp.linguisticdescription.sentence.flattenTokens
import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.deeplearning.transformers.BERT
import com.kotlinnlp.simplednn.deeplearning.transformers.BERTModel
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.DictionarySet
import com.kotlinnlp.utils.Serializer
import java.io.File
import java.io.FileInputStream

/**
 * Test a BERT model reconstructing the masked tokens in a text.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) {

  val parsedArgs = CommandLineArguments(args)

  val reconstructor = TextReconstructor(
    bert = BERT(model = parsedArgs.bertModelPath.let {
      println("Reading BERT model from '$it'...")
      BERTModel.load(FileInputStream(it))
    }),
    classifier = BERTModel.buildClassifier(params = parsedArgs.classifierModelPath.let {
      println("Reading classifier model from '$it'...")
      Serializer.deserialize(FileInputStream(File(it))) as FeedforwardLayerParameters
    }),
    dictionary = parsedArgs.dictionaryPath.let {
      println("Reading dictionary set from '$it'...")
      readDictionary(filename = parsedArgs.dictionaryPath, minOccurrences = 100, maxTerms = 20000)
    },
    tokenizer = NeuralTokenizer(model = parsedArgs.tokenizerModelPath.let {
      println("Reading tokenizer model from '$it'...")
      NeuralTokenizerModel.load(FileInputStream(it))
    }))

  while (true) {
    readInput()?.let {
      println("Reconstructed text: ${reconstructor.reconstruct(it)}")
    } ?: break
  }
}

/**
 * Reconstruct a text with masked words.
 *
 * @param tokenizer a text tokenizer
 * @param dictionary a dictionary with the possible forms to predict
 * @param bert a BERT transformer
 * @param classifier the terms classifier based on the [dictionary]
 */
private class TextReconstructor(
  val tokenizer: NeuralTokenizer,
  val dictionary: DictionarySet<String>,
  val bert: BERT,
  val classifier: FeedforwardLayer<DenseNDArray>
) {

  companion object {

    /**
     * The keyword that indicates the words to reconstruct in the input text.
     */
    const val RECONSTRUCT_KEY = "XXX"
  }

  /**
   * Reconstruct the masked words of a text.
   *
   * @param text the input text
   *
   * @return the given text with the [RECONSTRUCT_KEY]s replaced with the predictions of the [classifier]
   */
  fun reconstruct(text: String): String {

    val forms: List<String> = this.tokenizer.tokenize(text).flattenTokens().map { it.form }
    val embeddings: List<DenseNDArray> = forms.map { this.getEmbedding(it) }
    val encodings: List<DenseNDArray> = this.bert.forward(embeddings)

    val reconstructedForms: List<String> = forms.zip(encodings).map { (form, encoding) ->
      if (form == RECONSTRUCT_KEY) this.reconstructForm(encoding) else form
    }

    return reconstructedForms.joinToString(" ")
  }

  /**
   * @param form a token form
   *
   * @return the embedding vector that represents the given form
   */
  private fun getEmbedding(form: String): DenseNDArray = if (form == RECONSTRUCT_KEY)
    this.bert.model.embeddingsMap!!.unknownEmbedding.values
  else
    this.bert.model.embeddingsMap!![form].values

  /**
   * @param encoding the encoding of a token
   *
   * @return the token form reconstructed with the classifier
   */
  private fun reconstructForm(encoding: DenseNDArray): String {

    this.classifier.setInput(encoding)
    this.classifier.forward()

    val classification: DenseNDArray = this.classifier.outputArray.values

    return this.dictionary.getElement(id = classification.argMaxIndex())!!
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
