(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("faa6f597-4857-406c-a710-1cd6f3868210");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'faa6f597-4857-406c-a710-1cd6f3868210' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"851be1d5-581a-4095-ab93-082330abe9d2":{"roots":{"references":[{"attributes":{},"id":"39988","type":"BasicTickFormatter"},{"attributes":{"line_dash":[6],"location":0.41666666666666663},"id":"39920","type":"Span"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#c10c90"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39962","type":"VBar"},{"attributes":{},"id":"39979","type":"UnionRenderers"},{"attributes":{"below":[{"id":"39888"}],"center":[{"id":"39891"},{"id":"39895"},{"id":"39948"},{"id":"39954"},{"id":"39960"},{"id":"39966"}],"left":[{"id":"39892"}],"output_backend":"webgl","plot_height":331,"plot_width":496,"renderers":[{"id":"39946"},{"id":"39952"},{"id":"39958"},{"id":"39964"}],"title":{"id":"39969"},"toolbar":{"id":"39906"},"toolbar_location":null,"x_range":{"id":"39846"},"x_scale":{"id":"39884"},"y_range":{"id":"39848"},"y_scale":{"id":"39886"}},"id":"39881","subtype":"Figure","type":"Plot"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39923","type":"VBar"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#c10c90"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39935","type":"VBar"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39944","type":"VBar"},{"attributes":{"line_dash":[6],"location":1.4807692307692308},"id":"39954","type":"Span"},{"attributes":{"data":{"top":{"__ndarray__":"6YVe6IVe4D9nZmZmZmbeP2dmZmZmZu4/WWqlVmql7D/eyI3cyI3YP7vQC73QC9U/uBM7sRM73T+vEzuxEzvdPyZ2Yid2Ytc/lxu5kRu52T8ZuZEbuZHfP5AbuZEbudk/QS/0Qi/04D8LwQ/8wA/cP5AbuZEbudk/q9RKrdRK4z9BL/RCL/TgPyZ2Yid2Ytc/USu1Uiu1wj8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"39993"},"selection_policy":{"id":"39992"}},"id":"39943","type":"ColumnDataSource"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"39870","type":"BoxAnnotation"},{"attributes":{"data":{"top":{"__ndarray__":"MzMzMzOzDUC4HoXrUTgPQDCW/GLJrwxA0GkDnTbQDEBtoNMGOu0LQOi0gU4baApAzszMzMzMC0DrUbgehWsLQClcj8L1qApAqA102kAnCkBH4XoUrkcKQMaSXyz5xQlA6LSBThtoCkAqXI/C9agKQClcj8L1qApACtejcD0KC0AGOm2g0wYKQMkvlvxiyQpAaQOdNtDpCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"39984"},"selection_policy":{"id":"39983"}},"id":"39933","type":"ColumnDataSource"},{"attributes":{"source":{"id":"39933"}},"id":"39937","type":"CDSView"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39945","type":"VBar"},{"attributes":{"line_dash":[6],"location":0.48076923076923067},"id":"39948","type":"Span"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#328c06"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39956","type":"VBar"},{"attributes":{"source":{"id":"39915"}},"id":"39919","type":"CDSView"},{"attributes":{"data_source":{"id":"39943"},"glyph":{"id":"39944"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"39945"},"selection_glyph":null,"view":{"id":"39947"}},"id":"39946","type":"GlyphRenderer"},{"attributes":{},"id":"39996","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"39933"},"glyph":{"id":"39934"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"39935"},"selection_glyph":null,"view":{"id":"39937"}},"id":"39936","type":"GlyphRenderer"},{"attributes":{"ticks":[0,1,2,3]},"id":"39967","type":"FixedTicker"},{"attributes":{"text":"tau"},"id":"39941","type":"Title"},{"attributes":{"data_source":{"id":"39915"},"glyph":{"id":"39916"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"39917"},"selection_glyph":null,"view":{"id":"39919"}},"id":"39918","type":"GlyphRenderer"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"39896"},{"id":"39897"},{"id":"39898"},{"id":"39899"},{"id":"39900"},{"id":"39901"},{"id":"39902"},{"id":"39903"}]},"id":"39906","type":"Toolbar"},{"attributes":{"source":{"id":"39927"}},"id":"39931","type":"CDSView"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAA8D8OdNpApw30PxSuR+F6FPY/1AY6baDT9T8c6LSBThv4PxdLfrHkF/c/1QY6baDT9T+V/GLJL5b2P1jyiyW/WPc/43oUrkfh+T8ehetRuB75PxdLfrHkF/c/mJmZmZmZ9z8YrkfhehT2P1RVVVVVVfY/lfxiyS+W9j/gehSuR+H5P5iZmZmZmfc/kl8s+cWS9T8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"39980"},"selection_policy":{"id":"39979"}},"id":"39921","type":"ColumnDataSource"},{"attributes":{"line_dash":[6],"location":3.480769230769231},"id":"39966","type":"Span"},{"attributes":{},"id":"39977","type":"UnionRenderers"},{"attributes":{},"id":"39850","type":"LinearScale"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#328c06"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39957","type":"VBar"},{"attributes":{"axis_label":"Chain","formatter":{"id":"39974"},"ticker":{"id":"39939"}},"id":"39858","type":"LinearAxis"},{"attributes":{"source":{"id":"39943"}},"id":"39947","type":"CDSView"},{"attributes":{"source":{"id":"39955"}},"id":"39959","type":"CDSView"},{"attributes":{},"id":"39901","type":"UndoTool"},{"attributes":{"line_dash":[6],"location":2.4166666666666665},"id":"39932","type":"Span"},{"attributes":{"overlay":{"id":"39905"}},"id":"39900","type":"LassoSelectTool"},{"attributes":{},"id":"39989","type":"BasicTickFormatter"},{"attributes":{},"id":"39899","type":"WheelZoomTool"},{"attributes":{},"id":"39902","type":"SaveTool"},{"attributes":{},"id":"39978","type":"Selection"},{"attributes":{"source":{"id":"39921"}},"id":"39925","type":"CDSView"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#c10c90"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39934","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"4Qd+4Af+BUDVSq3USq0EQBh6oRd6oQJAGHqhF3qhAkAUO7ETOzECQCu1Uiu10gRAd2IndmKnA0DFTuzETuwCQHIjN3IjNwNAJDdyIzfyA0Bu5EZu5MYCQB/4gR/4gQNAxU7sxE7sAkDTC73QCz0EQNALvdALPQRA0Au90As9BEB6oRd6oRcEQIIf+IEf+ARAhl7ohV5oBUA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"39997"},"selection_policy":{"id":"39996"}},"id":"39955","type":"ColumnDataSource"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#328c06"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39929","type":"VBar"},{"attributes":{},"id":"39896","type":"ResetTool"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#fa7c17"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39922","type":"VBar"},{"attributes":{},"id":"39897","type":"PanTool"},{"attributes":{},"id":"39865","type":"WheelZoomTool"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#328c06"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39928","type":"VBar"},{"attributes":{"ticks":[0,1,2,3]},"id":"39939","type":"FixedTicker"},{"attributes":{"overlay":{"id":"39904"}},"id":"39898","type":"BoxZoomTool"},{"attributes":{"axis":{"id":"39858"},"dimension":1,"ticker":null},"id":"39861","type":"Grid"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAAAECkcD0K1yMBQCa/WPKLpQFA6LSBThtoAkBqA5020OkCQCz5xZJfrANA8O7u7u5uBEDrUbgehWsDQOtRuB6FawNAC9ejcD0KA0BKfrHkF0sDQA102kCnDQRAqqqqqqoqA0BQG+i0gU4EQC+W/GLJrwRAThvotIFOBECuR+F6FC4EQE4b6LSBTgRAcD0K16PwBEA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"39982"},"selection_policy":{"id":"39981"}},"id":"39927","type":"ColumnDataSource"},{"attributes":{"axis":{"id":"39892"},"dimension":1,"ticker":null},"id":"39895","type":"Grid"},{"attributes":{"data_source":{"id":"39921"},"glyph":{"id":"39922"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"39923"},"selection_glyph":null,"view":{"id":"39925"}},"id":"39924","type":"GlyphRenderer"},{"attributes":{"text":"mu"},"id":"39969","type":"Title"},{"attributes":{},"id":"39889","type":"BasicTicker"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"39988"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"39889"}},"id":"39888","type":"LinearAxis"},{"attributes":{},"id":"39974","type":"BasicTickFormatter"},{"attributes":{"axis_label":"Chain","formatter":{"id":"39989"},"ticker":{"id":"39967"}},"id":"39892","type":"LinearAxis"},{"attributes":{},"id":"39994","type":"UnionRenderers"},{"attributes":{"data":{"top":{"__ndarray__":"ZWZmZmZm7j9OG+i0gU7XP2cDnTbQad8/WfKLJb9Y2j9Bpw102kDTP17JL5b8Yt0/PW2g0wY60T9U8oslv1jaP1ws+cWSX9w/WlVVVVVV2T9SVVVVVVXZPzTQaQOdNuA/ZgOdNtBp3z9m8oslv1jaP0h+seQXS9Y/SH6x5BdL1j84baDTBjrRPz+nDXTaQNM/SH6x5BdL1j8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"39978"},"selection_policy":{"id":"39977"}},"id":"39915","type":"ColumnDataSource"},{"attributes":{"line_dash":[6],"location":3.4166666666666665},"id":"39938","type":"Span"},{"attributes":{"data_source":{"id":"39955"},"glyph":{"id":"39956"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"39957"},"selection_glyph":null,"view":{"id":"39959"}},"id":"39958","type":"GlyphRenderer"},{"attributes":{},"id":"39980","type":"Selection"},{"attributes":{"callback":null},"id":"39903","type":"HoverTool"},{"attributes":{},"id":"39884","type":"LinearScale"},{"attributes":{},"id":"39998","type":"UnionRenderers"},{"attributes":{"axis":{"id":"39888"},"ticker":null},"id":"39891","type":"Grid"},{"attributes":{},"id":"39981","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"39949"},"glyph":{"id":"39950"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"39951"},"selection_glyph":null,"view":{"id":"39953"}},"id":"39952","type":"GlyphRenderer"},{"attributes":{"data":{"top":{"__ndarray__":"EPzAD/zACUAg+IEf+IELQMEP/MAPfApAdmIndmKnC0A4ciM3ciMOQIZe6IVeaA1Ah17ohV5oDUDYiZ3YiR0NQD7wAz/wAw9Ae6EXeqEXDEAbuZEbuRELQHZiJ3ZipwtAeqEXeqEXDEB0IzdyIzcLQBu5kRu5EQtAFDuxEzsxCkByIzdyIzcLQBu5kRu5EQtAxU7sxE7sCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"39999"},"selection_policy":{"id":"39998"}},"id":"39961","type":"ColumnDataSource"},{"attributes":{"line_dash":[6],"location":2.480769230769231},"id":"39960","type":"Span"},{"attributes":{},"id":"39863","type":"PanTool"},{"attributes":{},"id":"39982","type":"Selection"},{"attributes":{},"id":"39993","type":"Selection"},{"attributes":{"toolbars":[{"id":"39872"},{"id":"39906"}],"tools":[{"id":"39862"},{"id":"39863"},{"id":"39864"},{"id":"39865"},{"id":"39866"},{"id":"39867"},{"id":"39868"},{"id":"39869"},{"id":"39896"},{"id":"39897"},{"id":"39898"},{"id":"39899"},{"id":"39900"},{"id":"39901"},{"id":"39902"},{"id":"39903"}]},"id":"40002","type":"ProxyToolbar"},{"attributes":{},"id":"39992","type":"UnionRenderers"},{"attributes":{"source":{"id":"39949"}},"id":"39953","type":"CDSView"},{"attributes":{},"id":"39867","type":"UndoTool"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"39905","type":"PolyAnnotation"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#fa7c17"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39950","type":"VBar"},{"attributes":{"below":[{"id":"39854"}],"center":[{"id":"39857"},{"id":"39861"},{"id":"39920"},{"id":"39926"},{"id":"39932"},{"id":"39938"}],"left":[{"id":"39858"}],"output_backend":"webgl","plot_height":331,"plot_width":496,"renderers":[{"id":"39918"},{"id":"39924"},{"id":"39930"},{"id":"39936"}],"title":{"id":"39941"},"toolbar":{"id":"39872"},"toolbar_location":null,"x_range":{"id":"39846"},"x_scale":{"id":"39850"},"y_range":{"id":"39848"},"y_scale":{"id":"39852"}},"id":"39845","subtype":"Figure","type":"Plot"},{"attributes":{"toolbar":{"id":"40002"},"toolbar_location":"above"},"id":"40003","type":"ToolbarBox"},{"attributes":{"data_source":{"id":"39927"},"glyph":{"id":"39928"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"39929"},"selection_glyph":null,"view":{"id":"39931"}},"id":"39930","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"39961"},"glyph":{"id":"39962"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"39963"},"selection_glyph":null,"view":{"id":"39965"}},"id":"39964","type":"GlyphRenderer"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"39904","type":"BoxAnnotation"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#c10c90"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39963","type":"VBar"},{"attributes":{},"id":"39999","type":"Selection"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39951","type":"VBar"},{"attributes":{},"id":"39997","type":"Selection"},{"attributes":{},"id":"39886","type":"LinearScale"},{"attributes":{"data":{"top":{"__ndarray__":"P/ADP/AD9z+SG7mRG7n2Py/0Qi/0QvU/eqEXeqEX9D9IbuRGbuT3P4If+IEf+PQ/MPRCL/RC9T+ZmZmZmZn3Pyd2Yid2YvQ/9kIv9EIv+D+4kRu5kRv7P7ATO7ETO/o/oBd6oRd6+D+mF3qhF3r4P1ZqpVZqpfk/9EIv9EIv+D9GbuRGbuT3P07sxE7sxPg/wA/8wA/8+z8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"39995"},"selection_policy":{"id":"39994"}},"id":"39949","type":"ColumnDataSource"},{"attributes":{},"id":"39848","type":"DataRange1d"},{"attributes":{"children":[{"id":"40003"},{"id":"40001"}]},"id":"40004","type":"Column"},{"attributes":{},"id":"39846","type":"DataRange1d"},{"attributes":{"source":{"id":"39961"}},"id":"39965","type":"CDSView"},{"attributes":{},"id":"39973","type":"BasicTickFormatter"},{"attributes":{},"id":"39868","type":"SaveTool"},{"attributes":{},"id":"39862","type":"ResetTool"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39917","type":"VBar"},{"attributes":{},"id":"39852","type":"LinearScale"},{"attributes":{},"id":"39984","type":"Selection"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"39871","type":"PolyAnnotation"},{"attributes":{"callback":null},"id":"39869","type":"HoverTool"},{"attributes":{},"id":"39855","type":"BasicTicker"},{"attributes":{},"id":"39983","type":"UnionRenderers"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"39973"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"39855"}},"id":"39854","type":"LinearAxis"},{"attributes":{"children":[[{"id":"39845"},0,0],[{"id":"39881"},0,1]]},"id":"40001","type":"GridBox"},{"attributes":{"overlay":{"id":"39871"}},"id":"39866","type":"LassoSelectTool"},{"attributes":{"axis":{"id":"39854"},"ticker":null},"id":"39857","type":"Grid"},{"attributes":{"line_dash":[6],"location":1.4166666666666665},"id":"39926","type":"Span"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"39916","type":"VBar"},{"attributes":{},"id":"39995","type":"Selection"},{"attributes":{"overlay":{"id":"39870"}},"id":"39864","type":"BoxZoomTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"39862"},{"id":"39863"},{"id":"39864"},{"id":"39865"},{"id":"39866"},{"id":"39867"},{"id":"39868"},{"id":"39869"}]},"id":"39872","type":"Toolbar"}],"root_ids":["40004"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"851be1d5-581a-4095-ab93-082330abe9d2","root_ids":["40004"],"roots":{"40004":"faa6f597-4857-406c-a710-1cd6f3868210"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();