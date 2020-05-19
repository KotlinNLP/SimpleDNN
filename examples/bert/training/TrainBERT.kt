/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package bert.training

import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.deeplearning.transformers.BERTModel
import com.kotlinnlp.simplednn.deeplearning.transformers.BERTTrainer
import com.kotlinnlp.utils.DictionarySet
import com.xenomachina.argparser.mainBody
import java.io.File

/**
 * Train a BERT model.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val vocabulary: DictionarySet<String> = parsedArgs.vocabularyPath.let {
    println("Reading vocabulary from '$it'...")
    readVocabulary(filename = parsedArgs.vocabularyPath, maxTerms = parsedArgs.maxVocabularySize)
  }

  val embeddingsMap: EmbeddingsMap<String> = parsedArgs.embeddingsPath?.let {
    println("Loading pre-trained word embeddings from '$it'...")
    EmbeddingsMap.load(it)
  } ?: EmbeddingsMap(size = 100)

  File(parsedArgs.datasetPath).useLines { examples ->

    println("Reading training set from '${parsedArgs.datasetPath}'...")

    val model = BERTModel(
      inputSize = embeddingsMap.size,
      attentionSize = embeddingsMap.size / 3,
      attentionOutputSize = embeddingsMap.size / 3,
      outputHiddenSize = 2048,
      numOfHeads = 3,
      numOfLayers = 3,
      vocabulary = vocabulary,
      wordEmbeddings = parsedArgs.embeddingsPath?.let { embeddingsMap })

    val helper = BERTTrainer(
      model = model,
      modelFilename = parsedArgs.modelPath,
      updateMethod = ADAMMethod(stepSize = 0.001),
      termsDropout = 0.15,
      optimizeEmbeddings = parsedArgs.embeddingsPath == null,
      examples = examples.asIterable(),
      shuffler = null,
      epochs = parsedArgs.epochs)

    println("\n-- Start training")
    helper.train()
  }
}

/**
 * Punctuation terms that are included automatically in the dictionary.
 */
private val punctTerms: List<String> = listOf(
  "’'", // apostrophe
  "()[]{}<>", // brackets
  ":", // colon
  ",", // comma
  "‒–—―", // dashes
  "…", // ellipsis
  "!", // exclamation mark
  ".", // full stop/period
  "«»", // guillemets
  "-‐", // hyphen
  "?", // question mark
  "‘’“”", // quotation marks
  ";", // semicolon
  "/", // slash/stroke
  "\\", // backslash
  "⁄", // solidus
  "␠", // space?
  "·", // interpunct
  "&", // ampersand
  "@", // at sign
  "*", // asterisk
  "•", // bullet
  "^", // caret
  "¤¢$€£¥₩₪", // currency
  "†‡", // dagger
  "°", // degree
  "¡", // inverted exclamation point
  "¿", // inverted question mark
  "¬", // negation
  "#", // number sign (hashtag)
  "№", // numero sign ()
  "%‰‱", // percent and related signs
  "¶", // pilcrow
  "′", // prime
  "§", // section sign
  "~", // tilde/swung dash
  "¨", // umlaut/diaeresis
  "_", // underscore/understrike
  "|¦", // vertical/pipe/broken bar
  "⁂", // asterism
  "☞", // index/fist
  "∴", // therefore sign
  "‽", // interrobang
  "※" // reference mark
)

/**
 * Read the vocabulary for the BERT training from file.
 * Each line of the file must contain a term.
 * On top of this, punctuation terms are inserted in the dictionary.
 *
 * @param filename the filename of the vocabulary
 * @param maxTerms the max number of terms to insert into the vocabulary or null for no limit
 *
 * @return a vocabulary for the training
 */
private fun readVocabulary(filename: String, maxTerms: Int? = null): DictionarySet<String> {

  val terms: List<String> = File(filename).readLines()

  return DictionarySet(terms.take(maxTerms ?: terms.size)).apply {
    punctTerms.forEach { it.forEach { c -> add(c.toString()) } }
  }
}
