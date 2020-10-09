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
    
      
      
    
      var element = document.getElementById("2b097954-35da-4211-9470-a593961bfc8b");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '2b097954-35da-4211-9470-a593961bfc8b' but no matching script tag was found.")
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
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
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
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
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
                    
                  var docs_json = '{"b3da6c93-a672-4840-a06c-77cfcd0b127e":{"roots":{"references":[{"attributes":{},"id":"26755","type":"LinearScale"},{"attributes":{},"id":"26787","type":"LinearScale"},{"attributes":{},"id":"26765","type":"ResetTool"},{"attributes":{},"id":"26766","type":"PanTool"},{"attributes":{"axis":{"id":"26791"},"ticker":null},"id":"26794","type":"Grid"},{"attributes":{"children":[[{"id":"26748"},0,0],[{"id":"26784"},0,1]]},"id":"26904","type":"GridBox"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26820","type":"VBar"},{"attributes":{},"id":"26897","type":"UnionRenderers"},{"attributes":{"line_dash":[6],"location":0.41666666666666663},"id":"26823","type":"Span"},{"attributes":{"ticks":[0,1,2,3]},"id":"26842","type":"FixedTicker"},{"attributes":{"axis":{"id":"26761"},"dimension":1,"ticker":null},"id":"26764","type":"Grid"},{"attributes":{"source":{"id":"26818"}},"id":"26822","type":"CDSView"},{"attributes":{},"id":"26898","type":"Selection"},{"attributes":{"below":[{"id":"26791"}],"center":[{"id":"26794"},{"id":"26798"},{"id":"26851"},{"id":"26857"},{"id":"26863"},{"id":"26869"}],"left":[{"id":"26795"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26849"},{"id":"26855"},{"id":"26861"},{"id":"26867"}],"title":{"id":"26872"},"toolbar":{"id":"26809"},"toolbar_location":null,"x_range":{"id":"26749"},"x_scale":{"id":"26787"},"y_range":{"id":"26751"},"y_scale":{"id":"26789"}},"id":"26784","subtype":"Figure","type":"Plot"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26826","type":"VBar"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26773","type":"BoxAnnotation"},{"attributes":{"data_source":{"id":"26818"},"glyph":{"id":"26819"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26820"},"selection_glyph":null,"view":{"id":"26822"}},"id":"26821","type":"GlyphRenderer"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26831","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAA8D8OdNpApw30PxSuR+F6FPY/1AY6baDT9T8c6LSBThv4PxdLfrHkF/c/1QY6baDT9T+V/GLJL5b2P1jyiyW/WPc/43oUrkfh+T8ehetRuB75PxdLfrHkF/c/mJmZmZmZ9z8YrkfhehT2P1RVVVVVVfY/lfxiyS+W9j/gehSuR+H5P5iZmZmZmfc/kl8s+cWS9T8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26881"},"selection_policy":{"id":"26880"}},"id":"26824","type":"ColumnDataSource"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26825","type":"VBar"},{"attributes":{"below":[{"id":"26757"}],"center":[{"id":"26760"},{"id":"26764"},{"id":"26823"},{"id":"26829"},{"id":"26835"},{"id":"26841"}],"left":[{"id":"26761"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26821"},{"id":"26827"},{"id":"26833"},{"id":"26839"}],"title":{"id":"26844"},"toolbar":{"id":"26775"},"toolbar_location":null,"x_range":{"id":"26749"},"x_scale":{"id":"26753"},"y_range":{"id":"26751"},"y_scale":{"id":"26755"}},"id":"26748","subtype":"Figure","type":"Plot"},{"attributes":{"source":{"id":"26824"}},"id":"26828","type":"CDSView"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26832","type":"VBar"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26837","type":"VBar"},{"attributes":{"callback":null},"id":"26806","type":"HoverTool"},{"attributes":{"data_source":{"id":"26824"},"glyph":{"id":"26825"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26826"},"selection_glyph":null,"view":{"id":"26828"}},"id":"26827","type":"GlyphRenderer"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26890"},"ticker":{"id":"26870"}},"id":"26795","type":"LinearAxis"},{"attributes":{},"id":"26899","type":"UnionRenderers"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26891"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26792"}},"id":"26791","type":"LinearAxis"},{"attributes":{"line_dash":[6],"location":1.4166666666666665},"id":"26829","type":"Span"},{"attributes":{},"id":"26792","type":"BasicTicker"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAAAECkcD0K1yMBQCa/WPKLpQFA6LSBThtoAkBqA5020OkCQCz5xZJfrANA8O7u7u5uBEDrUbgehWsDQOtRuB6FawNAC9ejcD0KA0BKfrHkF0sDQA102kCnDQRAqqqqqqoqA0BQG+i0gU4EQC+W/GLJrwRAThvotIFOBECuR+F6FC4EQE4b6LSBTgRAcD0K16PwBEA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26883"},"selection_policy":{"id":"26882"}},"id":"26830","type":"ColumnDataSource"},{"attributes":{},"id":"26900","type":"Selection"},{"attributes":{"source":{"id":"26830"}},"id":"26834","type":"CDSView"},{"attributes":{"overlay":{"id":"26774"}},"id":"26769","type":"LassoSelectTool"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26838","type":"VBar"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26807","type":"BoxAnnotation"},{"attributes":{"line_dash":[6],"location":0.48076923076923067},"id":"26851","type":"Span"},{"attributes":{"data_source":{"id":"26830"},"glyph":{"id":"26831"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26832"},"selection_glyph":null,"view":{"id":"26834"}},"id":"26833","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"26795"},"dimension":1,"ticker":null},"id":"26798","type":"Grid"},{"attributes":{"line_dash":[6],"location":2.4166666666666665},"id":"26835","type":"Span"},{"attributes":{},"id":"26890","type":"BasicTickFormatter"},{"attributes":{"data":{"top":{"__ndarray__":"ZWZmZmZm7j9OG+i0gU7XP2cDnTbQad8/WfKLJb9Y2j9Bpw102kDTP17JL5b8Yt0/PW2g0wY60T9U8oslv1jaP1ws+cWSX9w/WlVVVVVV2T9SVVVVVVXZPzTQaQOdNuA/ZgOdNtBp3z9m8oslv1jaP0h+seQXS9Y/SH6x5BdL1j84baDTBjrRPz+nDXTaQNM/SH6x5BdL1j8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26879"},"selection_policy":{"id":"26878"}},"id":"26818","type":"ColumnDataSource"},{"attributes":{},"id":"26878","type":"UnionRenderers"},{"attributes":{"data":{"top":{"__ndarray__":"MzMzMzOzDUC4HoXrUTgPQDCW/GLJrwxA0GkDnTbQDEBtoNMGOu0LQOi0gU4baApAzszMzMzMC0DrUbgehWsLQClcj8L1qApAqA102kAnCkBH4XoUrkcKQMaSXyz5xQlA6LSBThtoCkAqXI/C9agKQClcj8L1qApACtejcD0KC0AGOm2g0wYKQMkvlvxiyQpAaQOdNtDpCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26885"},"selection_policy":{"id":"26884"}},"id":"26836","type":"ColumnDataSource"},{"attributes":{"overlay":{"id":"26807"}},"id":"26801","type":"BoxZoomTool"},{"attributes":{"source":{"id":"26836"}},"id":"26840","type":"CDSView"},{"attributes":{},"id":"26800","type":"PanTool"},{"attributes":{},"id":"26879","type":"Selection"},{"attributes":{},"id":"26799","type":"ResetTool"},{"attributes":{"ticks":[0,1,2,3]},"id":"26870","type":"FixedTicker"},{"attributes":{},"id":"26805","type":"SaveTool"},{"attributes":{"data_source":{"id":"26836"},"glyph":{"id":"26837"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26838"},"selection_glyph":null,"view":{"id":"26840"}},"id":"26839","type":"GlyphRenderer"},{"attributes":{},"id":"26891","type":"BasicTickFormatter"},{"attributes":{},"id":"26770","type":"UndoTool"},{"attributes":{},"id":"26802","type":"WheelZoomTool"},{"attributes":{"line_dash":[6],"location":3.4166666666666665},"id":"26841","type":"Span"},{"attributes":{},"id":"26768","type":"WheelZoomTool"},{"attributes":{"overlay":{"id":"26808"}},"id":"26803","type":"LassoSelectTool"},{"attributes":{"source":{"id":"26846"}},"id":"26850","type":"CDSView"},{"attributes":{},"id":"26804","type":"UndoTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26799"},{"id":"26800"},{"id":"26801"},{"id":"26802"},{"id":"26803"},{"id":"26804"},{"id":"26805"},{"id":"26806"}]},"id":"26809","type":"Toolbar"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26848","type":"VBar"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26847","type":"VBar"},{"attributes":{},"id":"26880","type":"UnionRenderers"},{"attributes":{"data":{"top":{"__ndarray__":"6YVe6IVe4D9nZmZmZmbeP2dmZmZmZu4/WWqlVmql7D/eyI3cyI3YP7vQC73QC9U/uBM7sRM73T+vEzuxEzvdPyZ2Yid2Ytc/lxu5kRu52T8ZuZEbuZHfP5AbuZEbudk/QS/0Qi/04D8LwQ/8wA/cP5AbuZEbudk/q9RKrdRK4z9BL/RCL/TgPyZ2Yid2Ytc/USu1Uiu1wj8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26894"},"selection_policy":{"id":"26893"}},"id":"26846","type":"ColumnDataSource"},{"attributes":{"text":"mu"},"id":"26872","type":"Title"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26854","type":"VBar"},{"attributes":{},"id":"26881","type":"Selection"},{"attributes":{"data_source":{"id":"26846"},"glyph":{"id":"26847"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26848"},"selection_glyph":null,"view":{"id":"26850"}},"id":"26849","type":"GlyphRenderer"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26859","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"P/ADP/AD9z+SG7mRG7n2Py/0Qi/0QvU/eqEXeqEX9D9IbuRGbuT3P4If+IEf+PQ/MPRCL/RC9T+ZmZmZmZn3Pyd2Yid2YvQ/9kIv9EIv+D+4kRu5kRv7P7ATO7ETO/o/oBd6oRd6+D+mF3qhF3r4P1ZqpVZqpfk/9EIv9EIv+D9GbuRGbuT3P07sxE7sxPg/wA/8wA/8+z8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26896"},"selection_policy":{"id":"26895"}},"id":"26852","type":"ColumnDataSource"},{"attributes":{"toolbar":{"id":"26905"},"toolbar_location":"above"},"id":"26906","type":"ToolbarBox"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26853","type":"VBar"},{"attributes":{"source":{"id":"26852"}},"id":"26856","type":"CDSView"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26860","type":"VBar"},{"attributes":{},"id":"26896","type":"Selection"},{"attributes":{"toolbars":[{"id":"26775"},{"id":"26809"}],"tools":[{"id":"26765"},{"id":"26766"},{"id":"26767"},{"id":"26768"},{"id":"26769"},{"id":"26770"},{"id":"26771"},{"id":"26772"},{"id":"26799"},{"id":"26800"},{"id":"26801"},{"id":"26802"},{"id":"26803"},{"id":"26804"},{"id":"26805"},{"id":"26806"}]},"id":"26905","type":"ProxyToolbar"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26865","type":"VBar"},{"attributes":{"data_source":{"id":"26852"},"glyph":{"id":"26853"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26854"},"selection_glyph":null,"view":{"id":"26856"}},"id":"26855","type":"GlyphRenderer"},{"attributes":{"children":[{"id":"26906"},{"id":"26904"}]},"id":"26907","type":"Column"},{"attributes":{"line_dash":[6],"location":1.4807692307692308},"id":"26857","type":"Span"},{"attributes":{"overlay":{"id":"26773"}},"id":"26767","type":"BoxZoomTool"},{"attributes":{},"id":"26882","type":"UnionRenderers"},{"attributes":{"data":{"top":{"__ndarray__":"4Qd+4Af+BUDVSq3USq0EQBh6oRd6oQJAGHqhF3qhAkAUO7ETOzECQCu1Uiu10gRAd2IndmKnA0DFTuzETuwCQHIjN3IjNwNAJDdyIzfyA0Bu5EZu5MYCQB/4gR/4gQNAxU7sxE7sAkDTC73QCz0EQNALvdALPQRA0Au90As9BEB6oRd6oRcEQIIf+IEf+ARAhl7ohV5oBUA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26898"},"selection_policy":{"id":"26897"}},"id":"26858","type":"ColumnDataSource"},{"attributes":{"source":{"id":"26858"}},"id":"26862","type":"CDSView"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26774","type":"PolyAnnotation"},{"attributes":{},"id":"26883","type":"Selection"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26866","type":"VBar"},{"attributes":{"data_source":{"id":"26858"},"glyph":{"id":"26859"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26860"},"selection_glyph":null,"view":{"id":"26862"}},"id":"26861","type":"GlyphRenderer"},{"attributes":{"line_dash":[6],"location":2.480769230769231},"id":"26863","type":"Span"},{"attributes":{"data":{"top":{"__ndarray__":"EPzAD/zACUAg+IEf+IELQMEP/MAPfApAdmIndmKnC0A4ciM3ciMOQIZe6IVeaA1Ah17ohV5oDUDYiZ3YiR0NQD7wAz/wAw9Ae6EXeqEXDEAbuZEbuRELQHZiJ3ZipwtAeqEXeqEXDEB0IzdyIzcLQBu5kRu5EQtAFDuxEzsxCkByIzdyIzcLQBu5kRu5EQtAxU7sxE7sCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26900"},"selection_policy":{"id":"26899"}},"id":"26864","type":"ColumnDataSource"},{"attributes":{"source":{"id":"26864"}},"id":"26868","type":"CDSView"},{"attributes":{"text":"tau"},"id":"26844","type":"Title"},{"attributes":{"axis":{"id":"26757"},"ticker":null},"id":"26760","type":"Grid"},{"attributes":{},"id":"26893","type":"UnionRenderers"},{"attributes":{},"id":"26753","type":"LinearScale"},{"attributes":{"data_source":{"id":"26864"},"glyph":{"id":"26865"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26866"},"selection_glyph":null,"view":{"id":"26868"}},"id":"26867","type":"GlyphRenderer"},{"attributes":{},"id":"26771","type":"SaveTool"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26875"},"ticker":{"id":"26842"}},"id":"26761","type":"LinearAxis"},{"attributes":{},"id":"26894","type":"Selection"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26876"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26758"}},"id":"26757","type":"LinearAxis"},{"attributes":{},"id":"26884","type":"UnionRenderers"},{"attributes":{"line_dash":[6],"location":3.480769230769231},"id":"26869","type":"Span"},{"attributes":{},"id":"26749","type":"DataRange1d"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26808","type":"PolyAnnotation"},{"attributes":{},"id":"26875","type":"BasicTickFormatter"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26819","type":"VBar"},{"attributes":{},"id":"26885","type":"Selection"},{"attributes":{},"id":"26758","type":"BasicTicker"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26765"},{"id":"26766"},{"id":"26767"},{"id":"26768"},{"id":"26769"},{"id":"26770"},{"id":"26771"},{"id":"26772"}]},"id":"26775","type":"Toolbar"},{"attributes":{},"id":"26876","type":"BasicTickFormatter"},{"attributes":{"callback":null},"id":"26772","type":"HoverTool"},{"attributes":{},"id":"26789","type":"LinearScale"},{"attributes":{},"id":"26895","type":"UnionRenderers"},{"attributes":{},"id":"26751","type":"DataRange1d"}],"root_ids":["26907"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"b3da6c93-a672-4840-a06c-77cfcd0b127e","root_ids":["26907"],"roots":{"26907":"2b097954-35da-4211-9470-a593961bfc8b"}}];
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