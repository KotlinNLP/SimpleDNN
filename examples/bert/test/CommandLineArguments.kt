/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package bert.test

import com.xenomachina.argparser.ArgParser

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
   * The file path of the serialized BERT model.
   */
  val bertModelPath: String by parser.storing(
    "-b",
    "--bert",
    help="the file path of the serialized BERT model"
  )

  /**
   * The file path of output classifier model.
   */
  val classifierModelPath: String by parser.storing(
    "-c",
    "--classifier",
    help="the file path of output classifier model"
  )

  /**
   * The file path of the training dictionary.
   */
  val dictionaryPath: String by parser.storing(
    "-d",
    "--dictionary",
    help="the file path of the training dictionary"
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
   * Force parsing all arguments (only read ones are parsed by default).
   */
  init {
    this.parser.force()
  }
}
