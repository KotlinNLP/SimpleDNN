/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package bert.convert

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
   * The file path of the input model.
   */
  val inputModelPath: String by parser.storing(
    "-i",
    "--input",
    help="the file path of the input model"
  )

  /**
   * The path of the file in which to serialize the output model.
   */
  val outputModelPath: String by parser.storing(
    "-o",
    "--output",
    help="the path of the file in which to serialize the output model"
  )

  /**
   * The path of the vocabulary used to train the model.
   */
  val vocabPath: String by parser.storing(
    "-v",
    "--vocabulary",
    help="the path of the vocabulary used to train the model"
  )

  /**
   * The number of attention heads.
   */
  val numOfHeads: Int by parser.storing(
    "-a",
    "--attention-heads",
    help="the number of attention heads"
  ) { toInt() }

  /**
   * Force parsing all arguments (only read ones are parsed by default).
   */
  init {
    this.parser.force()
  }
}
