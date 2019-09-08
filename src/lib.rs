#![deny(missing_docs)]

/*!
This crate provides an implementation for calculating the positions of text
fragments to fit text withing a rectangle.

The `graphics` feature, which is on by default, allows the direct rendering of a fitted
text with the [`piston2d-graphics`](https://docs.rs/piston2d-graphics) crate.
*/

use std::{collections::HashMap, sync::Mutex};

#[cfg(feature = "graphics")]
use graphics::{character::CharacterCache, math::Matrix2d, Graphics, ImageSize, Transformed};
use once_cell::sync::Lazy;
use rusttype::{Error, Font, GlyphId, Scale};
use vector2math::*;

use crate::color::Color;

/// A way of indenting a line
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Indent {
    /// Indent with a certain number of spaces
    Space(usize),
    /// Indent with a number of points
    Point(f64),
}

impl Indent {
    /// Get the `Indent`'s width in points
    pub fn in_points<C>(self, cache: &mut C, font_size: u32) -> f64
    where
        C: CharacterWidthCache,
    {
        match self {
            Indent::Space(spaces) => cache.char_width(' ', font_size) * spaces as f64,
            Indent::Point(points) => points,
        }
    }
}

/// A list of positioned objects
///
/// `V` usually implements `Vector2`
pub type PositionedList<V, I> = Vec<(V, I)>;

/// Lines that have starting positions
///
/// `V` usually implements `Vector2`
pub type PositionedLines<V> = PositionedList<V, String>;

/// Lines with metadata that have starting positions
///
/// `V` usually implements `Vector2`
pub type PositionedLinesMeta<V, M> = PositionedList<V, (String, M)>;

/// A way of resizing text in a rectangle
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Resize {
    /// Make the text no larger than its original font size,
    /// but still try to fit it in the rectangle
    NoLarger,
    /// Make the text as large as possible while still
    /// fitting in the rectangle
    Max,
    /// Do not resize the text
    None,
}

impl Default for Resize {
    fn default() -> Self {
        Resize::NoLarger
    }
}

/// A format for some text
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TextFormat {
    /// The font size
    pub font_size: u32,
    /// The spacing between lines. This should usually be somewhere
    /// between `1.0` and `2.0`, but any scalar is valid
    pub line_spacing: f64,
    /// The number of spaces to indent the first line of a paragraph
    pub first_line_indent: Indent,
    /// The number of spaces to indent all lines of a paragraph
    /// after the first
    pub lines_indent: Indent,
    /// The color of the text
    pub color: Color,
    /// The resize strategy
    pub resize: Resize,
}

impl From<u32> for TextFormat {
    fn from(font_size: u32) -> Self {
        TextFormat::new(font_size)
    }
}

static DEFAULT_TEXT_FORMAT: Lazy<Mutex<TextFormat>> = Lazy::new(|| {
    Mutex::new(TextFormat {
        font_size: 30,
        line_spacing: 1.1,
        first_line_indent: Indent::Space(0),
        lines_indent: Indent::Space(0),
        color: color::WHITE,
        resize: Resize::NoLarger,
    })
});

impl Default for TextFormat {
    fn default() -> TextFormat {
        *DEFAULT_TEXT_FORMAT
            .lock()
            .expect("fit-text default TextFormat thread panicked")
    }
}

impl TextFormat {
    /// Create a default `TextFormat` with the given font size
    pub fn new(font_size: u32) -> TextFormat {
        TextFormat {
            font_size,
            ..Default::default()
        }
    }
    /// Use this `TextFormat` as the global default
    pub fn set_as_default(&self) {
        *DEFAULT_TEXT_FORMAT
            .lock()
            .expect("fit-text default TextFormat thread panicked") = *self;
    }
    /// Set the font size
    pub fn font_size(self, font_size: u32) -> Self {
        TextFormat { font_size, ..self }
    }
    /// Set the line spacing
    pub fn line_spacing(self, line_spacing: f64) -> Self {
        TextFormat {
            line_spacing,
            ..self
        }
    }
    /// Set the indentation of the first line
    pub fn first_line_indent(self, first_line_indent: Indent) -> Self {
        TextFormat {
            first_line_indent,
            ..self
        }
    }
    /// Set the indentation of all lines after the first
    pub fn lines_indent(self, lines_indent: Indent) -> Self {
        TextFormat {
            lines_indent,
            ..self
        }
    }
    /// Set the color
    pub fn color(self, color: Color) -> Self {
        TextFormat { color, ..self }
    }
    /// Set the resize strategy
    pub fn resize(self, resize: Resize) -> Self {
        TextFormat { resize, ..self }
    }
    /// Change the font size depending on the the resize strategy
    ///
    /// The given max size is not used if the strategy is `Resize::None`
    pub fn resize_font(self, max_size: u32) -> Self {
        TextFormat {
            font_size: match self.resize {
                Resize::NoLarger => self.font_size.min(max_size),
                Resize::Max => max_size,
                Resize::None => self.font_size,
            },
            ..self
        }
    }
}

/// Defines behavior of a cache of character widths.
///
/// In general, determining the width of a character glyph with a given font size
/// is a non-trivial calculation. Caching a width calculation for each character
/// and font size ensures that the calculation is only done once for each pair.
pub trait CharacterWidthCache: Sized {
    /// Get the width of a character at a font size
    fn char_width(&mut self, character: char, font_size: u32) -> f64;
    /// Get the width of a string at a font_size
    fn width(&mut self, text: &str, font_size: u32) -> f64 {
        text.chars().map(|c| self.char_width(c, font_size)).sum()
    }
    /// Split a string into a list of lines of text with the given format where no line
    /// is wider than the given max width. Newlines (`\n`) in the string are respected
    fn format_lines<S, F>(&mut self, text: S, max_width: f64, format: F) -> Vec<String>
    where
        S: AsRef<str>,
        F: Into<TextFormat>,
    {
        let format = format.into();
        let mut sized_lines = Vec::new();
        let mut first_line = true;
        // Iterate through lines
        for line in text.as_ref().lines() {
            // Initialize a result line
            let mut sized_line = String::new();
            // Apply the indentation
            let indent = if first_line {
                format.first_line_indent
            } else {
                format.lines_indent
            };
            let indent_width = indent.in_points(self, format.font_size);
            let mut curr_width = indent_width;
            // Iterate through words
            for word in line.split_whitespace() {
                // Get the word's width
                let width = self.width(word, format.font_size);
                // If the word goes past the max width...
                let fits_here = curr_width + width < max_width;
                let first_word_on_line = (curr_width - indent_width).abs() < f64::EPSILON;
                let fits_at_all = width < max_width;
                if !(fits_here || first_word_on_line && !fits_at_all) {
                    // Pop off the trailing space
                    sized_line.pop();
                    // Push the result line onto the result list
                    sized_lines.push(sized_line);
                    // Init next line
                    first_line = false;
                    sized_line = String::new();
                    // Apply the indentation
                    let indent = if first_line {
                        format.first_line_indent
                    } else {
                        format.lines_indent
                    };

                    let indent_width = indent.in_points(self, format.font_size);
                    curr_width = indent_width;
                }
                // Push the word onto the result line
                sized_line.push_str(word);
                sized_line.push(' ');
                curr_width = curr_width + width + self.char_width(' ', format.font_size);
            }
            // Push the result line onto the result list
            sized_line.pop();
            sized_lines.push(sized_line);
            first_line = false;
        }
        sized_lines
    }
    /// Get the width of the widest line after performing
    /// the calculation of `CharacterWidthCache::format_lines`
    fn max_line_width<S, F>(&mut self, text: S, max_width: f64, format: F) -> f64
    where
        S: AsRef<str>,
        F: Into<TextFormat>,
    {
        let format = format.into();
        let lines = self.format_lines(text, max_width, format);
        lines
            .into_iter()
            .map(|line| self.width(&line, format.font_size))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }
    /// Calculate a set of positioned lines of text with the given format
    /// that fit within the given rectangle
    fn justify_text<S, R, F>(&mut self, text: S, rect: R, format: F) -> PositionedLines<R::Vector>
    where
        S: AsRef<str>,
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        let format = format.into();
        self.format_lines(text, rect.width(), format)
            .into_iter()
            .enumerate()
            .map(|(i, line)| {
                let y_offset = rect.top()
                    + f64::from(format.font_size)
                    + i as f64 * f64::from(format.font_size) * format.line_spacing;
                let x_offset = rect.left()
                    + if i == 0 {
                        format.first_line_indent.in_points(self, format.font_size)
                    } else {
                        format.lines_indent.in_points(self, format.font_size)
                    };
                (R::Vector::new(x_offset, y_offset), line)
            })
            .collect()
    }
    /// Calculate a set of positioned text and metadata lines with the given format
    /// that fit within the given rectangle
    ///
    /// This is useful when you have multiple fragments of text each with some
    /// associated metadata, such as a color. Fragments that are split between
    /// lines will have their metadata cloned
    fn justify_meta_fragments<I, M, S, R, F>(
        &mut self,
        fragments: I,
        rect: R,
        format: F,
    ) -> PositionedLinesMeta<R::Vector, M>
    where
        I: IntoIterator<Item = (S, M)>,
        M: Clone,
        S: AsRef<str>,
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        let format = format.into();
        let mut plm = Vec::new();
        let mut vert = 0.0;
        let space_width = self.char_width(' ', format.font_size);
        let mut horiz = format.first_line_indent.in_points(self, format.font_size);
        // Iterate through fragments
        for (text, meta) in fragments {
            let sub_rect = R::new(
                R::Vector::new(rect.left(), rect.top() + vert),
                R::Vector::new(rect.width(), rect.height() - vert),
            );
            let sub_format = format.first_line_indent(Indent::Point(horiz));
            let positioned_lines = self.justify_text(text, sub_rect, sub_format);
            horiz = match positioned_lines.len() {
                0 => 0.0,
                1 => horiz + self.width(&positioned_lines[0].1, format.font_size),
                _ => self.width(&positioned_lines.last().unwrap().1, format.font_size),
            } + space_width;
            if !positioned_lines.is_empty() {
                vert += (positioned_lines.len() - 1) as f64
                    * f64::from(format.font_size)
                    * format.line_spacing;
            }
            plm.extend(
                positioned_lines
                    .into_iter()
                    .map(|(v, l)| (v, (l, meta.clone()))),
            );
        }
        plm
    }
    /// Check if text with the given format fits within a rectangle's width
    fn text_fits_horizontal<R, F>(&mut self, text: &str, rect: R, format: F) -> bool
    where
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        self.max_line_width(text, rect.width(), format) < rect.width()
    }
    /// Check if text with the given format fits within a rectangle's height
    fn text_fits_vertical<R, F>(&mut self, text: &str, rect: R, format: F) -> bool
    where
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        let format = format.into();
        let lines = self.format_lines(text, rect.width(), format);
        if lines.is_empty() {
            return true;
        }
        let last_line_y = rect.top()
            + f64::from(format.font_size)
            + (lines.len() - 1) as f64 * f64::from(format.font_size) * format.line_spacing;
        last_line_y < rect.bottom()
    }
    /// Check if text with the given format fits within a rectangle
    fn text_fits<R, F>(&mut self, text: &str, rect: R, format: F) -> bool
    where
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        let format = format.into();
        self.text_fits_horizontal(text, rect, format) && self.text_fits_vertical(text, rect, format)
    }
    /// Determine the maximum font size for text with the given format
    /// that will still allow the text to fit within a rectangle
    fn fit_max_font_size<R, F>(&mut self, text: &str, rect: R, format: F) -> u32
    where
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        let mut format = format.into();
        while !self.text_fits(text, rect, format) {
            format.font_size -= 1;
        }
        format.font_size
    }
    /// Determine the minumum height for a rectangle such that text
    /// with the given format will still fit within the rectangle
    ///
    /// The given delta value defines how much to increment the
    /// rectangle's height on each check. Lower deltas will yield
    /// more accurate results, but will take longer to computer.
    fn fit_min_height<R, F>(&mut self, text: &str, mut rect: R, format: F, delta: f64) -> f64
    where
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        let format = format.into();
        let delta = delta.abs().max(1.0);
        while self.text_fits_vertical(text, rect, format) {
            rect = rect.with_size(R::Vector::new(rect.width(), rect.height() - delta))
        }
        while !self.text_fits_vertical(text, rect, format) {
            rect = rect.with_size(R::Vector::new(rect.width(), rect.height() + delta))
        }
        rect.height()
    }
    /// Determine the minumum width for a rectangle such that text
    /// with the given format will still fit within the rectangle
    ///
    /// The given delta value defines how much to increment the
    /// rectangle's width on each check. Lower deltas will yield
    /// more accurate results, but will take longer to computer.
    fn fit_min_width<R, F>(&mut self, text: &str, mut rect: R, format: F, delta: f64) -> f64
    where
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        let format = format.into();
        let delta = delta.abs().max(1.0);
        while self.text_fits(text, rect, format) {
            rect = rect.with_size(R::Vector::new(rect.width() - delta, rect.height()))
        }
        while !self.text_fits(text, rect, format) {
            rect = rect.with_size(R::Vector::new(rect.width() + delta, rect.height()))
        }
        rect.width()
    }
    /// Determine the correct text size based on the given `TextFormat`
    fn ideal_text_size<R, F>(&mut self, text: &str, rect: R, format: F) -> TextFormat
    where
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        let mut format = format.into();
        match format.resize {
            Resize::None => {}
            Resize::NoLarger => {
                while format.font_size > 0 && !self.text_fits(text, rect, format) {
                    format.font_size -= 1;
                }
            }
            Resize::Max => {
                while format.font_size > 0 && !self.text_fits(text, rect, format) {
                    format.font_size -= 1;
                }
                while self.text_fits(text, rect, format) {
                    format.font_size += 1;
                }
                format.font_size -= 1;
            }
        }
        format
    }
}

/// A basic implementor for `CharacterWidthCache`
#[derive(Clone)]
pub struct BasicGlyphs<'f> {
    widths: HashMap<(u32, char), f64>,
    font: Font<'f>,
}

impl<'f> BasicGlyphs<'f> {
    /// Loads a `Glyphs` from an array of font data.
    pub fn from_bytes(bytes: &'f [u8]) -> Result<BasicGlyphs<'f>, Error> {
        Ok(BasicGlyphs {
            widths: HashMap::new(),
            font: Font::from_bytes(bytes)?,
        })
    }
    /// Loads a `Glyphs` from a `Font`.
    pub fn from_font(font: Font<'f>) -> BasicGlyphs<'f> {
        BasicGlyphs {
            widths: HashMap::new(),
            font,
        }
    }
}

impl<'f> CharacterWidthCache for BasicGlyphs<'f> {
    fn char_width(&mut self, character: char, font_size: u32) -> f64 {
        let font = &self.font;
        *self
            .widths
            .entry((font_size, character))
            .or_insert_with(|| {
                let scale = Scale::uniform(font_size as f32);
                let glyph = font.glyph(character).scaled(scale);
                let glyph = if glyph.id() == GlyphId(0) && glyph.shape().is_none() {
                    font.glyph('\u{FFFD}').scaled(scale)
                } else {
                    glyph
                };
                glyph.h_metrics().advance_width.into()
            })
    }
}

#[cfg(feature = "graphics")]
impl<C> CharacterWidthCache for C
where
    C: CharacterCache,
{
    fn char_width(&mut self, character: char, font_size: u32) -> f64 {
        if let Ok(character) = <Self as CharacterCache>::character(self, font_size, character) {
            character.advance_size.x()
        } else {
            panic!("CharacterWidthCache::character returned Err")
        }
    }
}

/// Draw justified text to something using the `piston2d-graphics` crate
///
/// Text will be drawn in the given rectangle and use the given format
#[cfg(feature = "graphics")]
pub fn fitted_text<S, R, F, C, G>(
    text: S,
    rect: R,
    format: F,
    glyphs: &mut C,
    transform: Matrix2d,
    graphics: &mut G,
) -> Result<(), C::Error>
where
    S: AsRef<str>,
    R: Rectangle<Scalar = f64>,
    F: Into<TextFormat>,
    C: CharacterCache,
    C::Texture: ImageSize,
    G: Graphics<Texture = C::Texture>,
{
    let format = glyphs.ideal_text_size(text.as_ref(), rect, format);
    for (pos, line) in glyphs.justify_text(text.as_ref(), rect, format) {
        graphics::text(
            format.color,
            format.font_size,
            &line,
            glyphs,
            transform.trans(pos.x(), pos.y()),
            graphics,
        )?;
    }
    Ok(())
}

/// Draw justified, colored text fragments to something using the `piston2d-graphics` crate
///
/// Text will be drawn in the given rectangle and use the given format
#[cfg(feature = "graphics")]
pub fn fitted_colored_text<I, S, R, F, C, G>(
    fragments: I,
    rect: R,
    format: F,
    glyphs: &mut C,
    transform: Matrix2d,
    graphics: &mut G,
) -> Result<(), C::Error>
where
    I: IntoIterator<Item = (S, Color)>,
    S: AsRef<str>,
    R: Rectangle<Scalar = f64>,
    F: Into<TextFormat>,
    C: CharacterCache,
    C::Texture: ImageSize,
    G: Graphics<Texture = C::Texture>,
{
    let fragments: Vec<_> = fragments.into_iter().collect();
    let whole_string: String = fragments.iter().map(|(s, _)| s.as_ref()).collect();
    let format = glyphs.ideal_text_size(whole_string.as_ref(), rect, format);
    for (pos, (fragment, color)) in glyphs.justify_meta_fragments(fragments, rect, format) {
        graphics::text(
            color,
            format.font_size,
            &fragment,
            glyphs,
            transform.trans(pos.x(), pos.y()),
            graphics,
        )?;
    }
    Ok(())
}

/// A struct for writing text into multiple rectangles
#[cfg(feature = "graphics")]
pub struct Scribe<'a, C, G> {
    /// The text format
    pub format: TextFormat,
    /// The character cache
    pub glyphs: &'a mut C,
    /// The transform
    pub transform: Matrix2d,
    /// The graphics abstraction being drawn to
    pub graphics: &'a mut G,
}

#[cfg(feature = "graphics")]
impl<'a, C, G> Scribe<'a, C, G> {
    /// Create a new `Scribe`
    pub fn new(
        format: TextFormat,
        glyphs: &'a mut C,
        transform: Matrix2d,
        graphics: &'a mut G,
    ) -> Self {
        Scribe {
            format,
            glyphs,
            transform,
            graphics,
        }
    }
}

#[cfg(feature = "graphics")]
impl<'a, C, G> Scribe<'a, C, G>
where
    C: CharacterCache,
    G: Graphics<Texture = C::Texture>,
{
    /// Write some text into a rectangle with the `Scribe`
    pub fn write<S, R>(&mut self, text: S, rectangle: R) -> Result<(), C::Error>
    where
        S: AsRef<str>,
        R: Rectangle<Scalar = f64>,
    {
        fitted_text(
            text.as_ref(),
            rectangle,
            self.format,
            self.glyphs,
            self.transform,
            self.graphics,
        )
    }
    /// Write some colored text fragments into a rectangle with the `Scribe`
    pub fn write_colored<I, S, R>(&mut self, fragments: I, rectangle: R) -> Result<(), C::Error>
    where
        I: IntoIterator<Item = (S, Color)>,
        S: AsRef<str>,
        R: Rectangle<Scalar = f64>,
    {
        fitted_colored_text(
            fragments,
            rectangle,
            self.format,
            self.glyphs,
            self.transform,
            self.graphics,
        )
    }
}

/// Defines serveral color constants
pub mod color {
    /// An RGBA color
    pub type Color = [f32; 4];

    /// Red
    pub const RED: Color = [1.0, 0.0, 0.0, 1.0];
    /// Orange
    pub const ORANGE: Color = [1.0, 0.5, 0.0, 1.0];
    /// Yellow
    pub const YELLOW: Color = [1.0, 1.0, 0.0, 1.0];
    /// Green
    pub const GREEN: Color = [0.0, 1.0, 0.0, 1.0];
    /// Cyan
    pub const CYAN: Color = [0.0, 1.0, 1.0, 1.0];
    /// Blue
    pub const BLUE: Color = [0.0, 0.0, 1.0, 1.0];
    /// Purple
    pub const PURPLE: Color = [0.5, 0.0, 0.5, 1.0];
    /// Magenta
    pub const MAGENTA: Color = [1.0, 0.0, 1.0, 1.0];
    /// Black
    pub const BLACK: Color = [0.0, 0.0, 0.0, 1.0];
    /// Gray (same as `GREY`)
    pub const GRAY: Color = [0.5, 0.5, 0.5, 1.0];
    /// Grey (same as `GRAY`)
    pub const GREY: Color = GRAY;
    /// White
    pub const WHITE: Color = [1.0; 4];
    /// Transparent
    pub const TRANSPARENT: Color = [0.0; 4];
}
