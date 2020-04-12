/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package bert

import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.deeplearning.transformers.BERTParameters
import com.kotlinnlp.simplednn.deeplearning.transformers.BERTTrainer
import com.kotlinnlp.utils.DictionarySet
import java.io.File
import java.io.FileInputStream

/**
 * Execute the training of a BERT model.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) {

  val parsedArgs = CommandLineArguments(args)

  val tokenizer = NeuralTokenizer(model = parsedArgs.tokenizerModelPath.let {
    println("Reading tokenizer model from '$it'...")
    NeuralTokenizerModel.load(FileInputStream(it))
  })

  val dictionary: DictionarySet<String> = parsedArgs.dictionaryPath.let {
    println("Reading dictionary set from '$it'...")
    readDictionary(filename = parsedArgs.dictionaryPath, minOccurrences = 100, maxTerms = 20000)
  }

  val embeddingsMap: EmbeddingsMap<String> = parsedArgs.embeddingsPath.let {
    println("Loading pre-trained word embeddings from '$it'...")
    EmbeddingsMap.load(it)
  }

  File(parsedArgs.datasetPath).useLines { examples ->

    println("Reading training set from '${parsedArgs.datasetPath}'...")

    val model = BERTParameters(
      inputSize = embeddingsMap.size,
      attentionSize = 100,
      hiddenSize = 100,
      multiHeadStack = 3,
      dropout = 0.15)

    val helper = BERTTrainer(
      model = model,
      modelFilename = parsedArgs.modelPath,
      tokenizer = tokenizer,
      embeddingsMap = embeddingsMap,
      dictionary = dictionary,
      updateMethod = ADAMMethod(stepSize = 0.001),
      termsDropout = 0.15,
      examples = examples.asIterable(),
      shuffler = null,
      epochs = parsedArgs.epochs)

    println("\n-- Start training")
    helper.train()
  }
}

/**
 * Read the dictionary set for the training from file.
 * Each line of the file must contain a term and its occurrences, separated by a tab char (`\t`).
 *
 * @param filename the filename of the dictionary
 * @param minOccurrences the min number of occurrences to insert a term into the dictionary
 * @param maxTerms the max number of terms to insert into the dictionary or null for no limit
 *
 * @return a dictionary set for the training
 */
private fun readDictionary(filename: String, minOccurrences: Int, maxTerms: Int? = null): DictionarySet<String> {

  val terms: List<String> = File(filename)
    .readLines()
    .asSequence()
    .map { it.split("\t") }
    .map { it[0] to it[1].toInt() }
    .filter { it.second >= minOccurrences }
    .sortedByDescending { it.second }
    .map { it.first }
    .toList()

  return DictionarySet(terms.take(maxTerms ?: terms.size))
}
