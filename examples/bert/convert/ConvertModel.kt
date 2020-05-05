/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package bert.convert

import com.kotlinnlp.simplednn.deeplearning.transformers.BERTBaseImportHelper
import com.kotlinnlp.simplednn.deeplearning.transformers.BERTModel
import com.kotlinnlp.utils.DictionarySet
import java.io.File
import java.io.FileOutputStream

/**
 * Build a [BERTModel] from a file of named parameters and serialize it to file.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) {

  val parsedArgs = CommandLineArguments(args)

  val bertModel = BERTBaseImportHelper.buildModel(
    params = parsedArgs.inputModelPath.let {
      println("Reading parameters from '$it'...")
      BERTBaseImportHelper.readParams(filename = it, numOfHeads = parsedArgs.numOfHeads)
    },
    vocab = parsedArgs.vocabPath.let {
      println("Reading vocabulary from '$it'...")
      DictionarySet<String>().apply { File(it).forEachLine { line -> add(line.trim()) } }
    },
    numOfHeads = parsedArgs.numOfHeads)

  parsedArgs.outputModelPath.let {
    println("Serializing the model to '$it'...")
    bertModel.dump(FileOutputStream(File(it)))
  }
}
