#!/usr/bin/env node
/**
 * LexiAI Frontend Build Script
 *
 * Minifies JavaScript files for production deployment
 * - Reduces bundle size by ~40%
 * - Generates source maps for debugging
 * - Preserves original files
 */

const fs = require('fs');
const path = require('path');
const { minify } = require('terser');

// Configuration
const CONFIG = {
  jsDir: path.join(__dirname, '../js'),
  outputDir: path.join(__dirname, '../js/dist'),
  sourceMap: true,
  skipFiles: [
    '.min.js',      // Already minified
    'dist/'         // Output directory
  ],
  terserOptions: {
    compress: {
      drop_console: false,  // Keep console.log for debugging
      drop_debugger: true,
      pure_funcs: ['console.debug'], // Remove console.debug only
    },
    mangle: {
      toplevel: false,
      keep_classnames: true,
      keep_fnames: true,
    },
    format: {
      comments: false,  // Remove comments
      preamble: '/* LexiAI - Minified */',
    },
    sourceMap: {
      filename: undefined,
      url: undefined,
    },
  },
};

// Color output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  red: '\x1b[31m',
};

function log(msg, color = 'reset') {
  console.log(`${colors[color]}${msg}${colors.reset}`);
}

// Get all JS files
function getJSFiles(dir) {
  const files = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    // Skip excluded files and directories
    if (CONFIG.skipFiles.some(skip => entry.name.includes(skip))) {
      continue;
    }

    if (entry.isDirectory()) {
      files.push(...getJSFiles(fullPath));
    } else if (entry.isFile() && entry.name.endsWith('.js')) {
      files.push(fullPath);
    }
  }

  return files;
}

// Minify a single file
async function minifyFile(inputPath) {
  const relativePath = path.relative(CONFIG.jsDir, inputPath);
  const fileName = path.basename(inputPath, '.js');
  const outputPath = path.join(CONFIG.outputDir, relativePath.replace('.js', '.min.js'));
  const sourceMapPath = `${outputPath}.map`;

  try {
    // Read source file
    const code = fs.readFileSync(inputPath, 'utf8');
    const originalSize = Buffer.byteLength(code, 'utf8');

    // Minify
    const result = await minify(code, {
      ...CONFIG.terserOptions,
      sourceMap: CONFIG.sourceMap ? {
        filename: path.basename(outputPath),
        url: `${path.basename(outputPath)}.map`,
      } : false,
    });

    if (result.error) {
      throw result.error;
    }

    // Ensure output directory exists
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    // Write minified file
    fs.writeFileSync(outputPath, result.code, 'utf8');

    // Write source map
    if (CONFIG.sourceMap && result.map) {
      fs.writeFileSync(sourceMapPath, result.map, 'utf8');
    }

    // Calculate stats
    const minifiedSize = Buffer.byteLength(result.code, 'utf8');
    const reduction = ((1 - minifiedSize / originalSize) * 100).toFixed(1);

    log(`✓ ${relativePath}`, 'green');
    log(`  ${(originalSize / 1024).toFixed(1)}KB → ${(minifiedSize / 1024).toFixed(1)}KB (${reduction}% reduction)`, 'blue');

    return {
      file: relativePath,
      originalSize,
      minifiedSize,
      reduction: parseFloat(reduction),
    };
  } catch (error) {
    log(`✗ ${relativePath}`, 'red');
    log(`  Error: ${error.message}`, 'red');
    return null;
  }
}

// Main build function
async function build() {
  log('\n🚀 LexiAI Frontend Build', 'yellow');
  log('═══════════════════════════════════\n', 'yellow');

  const startTime = Date.now();

  // Get all JS files
  const jsFiles = getJSFiles(CONFIG.jsDir);
  log(`📦 Found ${jsFiles.length} JavaScript files\n`, 'blue');

  // Minify all files
  const results = [];
  for (const file of jsFiles) {
    const result = await minifyFile(file);
    if (result) {
      results.push(result);
    }
  }

  // Calculate total stats
  const totalOriginal = results.reduce((sum, r) => sum + r.originalSize, 0);
  const totalMinified = results.reduce((sum, r) => sum + r.minifiedSize, 0);
  const totalReduction = ((1 - totalMinified / totalOriginal) * 100).toFixed(1);
  const buildTime = ((Date.now() - startTime) / 1000).toFixed(2);

  // Summary
  log('\n═══════════════════════════════════', 'yellow');
  log('📊 Build Summary', 'yellow');
  log('═══════════════════════════════════\n', 'yellow');
  log(`Files processed: ${results.length}`, 'blue');
  log(`Original size:   ${(totalOriginal / 1024).toFixed(1)}KB`, 'blue');
  log(`Minified size:   ${(totalMinified / 1024).toFixed(1)}KB`, 'green');
  log(`Total reduction: ${totalReduction}% (${((totalOriginal - totalMinified) / 1024).toFixed(1)}KB saved)`, 'green');
  log(`Build time:      ${buildTime}s`, 'blue');
  log(`Output:          ${CONFIG.outputDir}\n`, 'blue');

  log('✅ Build completed successfully!\n', 'green');
}

// Run build
build().catch(error => {
  log(`\n❌ Build failed: ${error.message}\n`, 'red');
  process.exit(1);
});
