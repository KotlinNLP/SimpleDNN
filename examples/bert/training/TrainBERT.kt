/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package bert.training

import bert.readDictionary
import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.deeplearning.transformers.BERTModel
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
      multiHeadStack = 3,
      numOfLayers = 3,
      dropout = 0.15)

    if (parsedArgs.embeddingsPath == null)
      model.embeddingsMap = embeddingsMap

    val helper = BERTTrainer(
      model = model,
      modelFilename = parsedArgs.modelPath,
      classifierModelFilename = parsedArgs.classifierModelPath,
      tokenizer = tokenizer,
      embeddingsMap = embeddingsMap,
      dictionary = dictionary,
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
