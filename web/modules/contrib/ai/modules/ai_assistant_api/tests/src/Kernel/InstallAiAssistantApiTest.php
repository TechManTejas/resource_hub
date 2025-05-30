<?php

namespace Drupal\Tests\ai_assistant_api\Kernel;

use Drupal\KernelTests\KernelTestBase;

/**
 * Tests enabling ai_assistant_api and its dependencies.
 *
 * @group ai_assistant_api
 */
class InstallAiAssistantApiTest extends KernelTestBase {

  /**
   * Modules to enable before running the tests.
   *
   * @var array
   */
  protected static $modules = ['system', 'user', 'ai'];

  /**
   * Tests if the module installs successfully.
   */
  public function testModuleCanBeEnabled() {

    try {
      // Try to enable the module.
      \Drupal::service('module_installer')->install(['ai_assistant_api']);
      $this->assertTrue(\Drupal::service('module_handler')->moduleExists('ai_assistant_api'), 'The module is successfully installed.');
    }
    catch (\Exception $e) {
      $this->fail('The module could not be enabled: ' . $e->getMessage());
    }
  }

}
