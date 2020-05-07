/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package bert.training

import com.xenomachina.argparser.ArgParser
import com.xenomachina.argparser.default

/**
 * The interpreter of command line arguments.
 *
 * @param args the array of command line arguments
 */
internal class CommandLineArguments(args: Array<String>) {

  /**
   * The parser of the string arguments.
   */
  private val parser = ArgParser(args)

  /**
   * The number of training epochs (default = 1).
   */
  val epochs: Int by parser.storing(
    "-e",
    "--epochs",
    help="the number of training epochs (default = 1)"
  ) { toInt() }.default(1)

  /**
   * The file path in which to serialize the model.
   */
  val modelPath: String by parser.storing(
    "-m",
    "--model-path",
    help="the file path in which to serialize the model"
  )

  /**
   * The file path of the serialized tokenizer model.
   */
  val tokenizerModelPath: String by parser.storing(
    "-t",
    "--tokenizer",
    help="the file path of the serialized tokenizer model"
  )

  /**
   * The file path of the pre-trained word embeddings.
   */
  val embeddingsPath: String? by parser.storing(
    "-w",
    "--word-embeddings",
    help="the file path of the pre-trained word embeddings"
  ).default { null }

  /**
   * The file path of the training dataset.
   */
  val datasetPath: String by parser.storing(
    "-d",
    "--dataset",
    help="the file path of the training dataset"
  )

  /**
   * The file path of the training vocabulary.
   */
  val vocabularyPath: String by parser.storing(
    "-v",
    "--vocabulary",
    help="the file path of the training vocabulary"
  )

  /**
   * Force parsing all arguments (only read ones are parsed by default).
   */
  init {
    this.parser.force()
  }
}
