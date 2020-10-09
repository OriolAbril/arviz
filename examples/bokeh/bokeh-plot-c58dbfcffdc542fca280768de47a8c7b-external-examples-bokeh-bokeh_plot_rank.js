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
    
      
      
    
      var element = document.getElementById("3a519593-e510-4232-9be0-017895eb94ba");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '3a519593-e510-4232-9be0-017895eb94ba' but no matching script tag was found.")
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
                    
                  var docs_json = '{"75e54a11-f7ee-472f-b73e-c156bd6aa06f":{"roots":{"references":[{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26815","type":"VBar"},{"attributes":{},"id":"26744","type":"LinearScale"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26820","type":"VBar"},{"attributes":{},"id":"26704","type":"DataRange1d"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26814","type":"VBar"},{"attributes":{"children":[{"id":"26861"},{"id":"26859"}]},"id":"26862","type":"Column"},{"attributes":{"data":{"top":{"__ndarray__":"4Qd+4Af+BUDVSq3USq0EQBh6oRd6oQJAGHqhF3qhAkAUO7ETOzECQCu1Uiu10gRAd2IndmKnA0DFTuzETuwCQHIjN3IjNwNAJDdyIzfyA0Bu5EZu5MYCQB/4gR/4gQNAxU7sxE7sAkDTC73QCz0EQNALvdALPQRA0Au90As9BEB6oRd6oRcEQIIf+IEf+ARAhl7ohV5oBUA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26853"},"selection_policy":{"id":"26852"}},"id":"26813","type":"ColumnDataSource"},{"attributes":{},"id":"26846","type":"BasicTickFormatter"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26762","type":"BoxAnnotation"},{"attributes":{},"id":"26837","type":"UnionRenderers"},{"attributes":{"source":{"id":"26813"}},"id":"26817","type":"CDSView"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26821","type":"VBar"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26728","type":"BoxAnnotation"},{"attributes":{},"id":"26838","type":"Selection"},{"attributes":{"data_source":{"id":"26813"},"glyph":{"id":"26814"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26815"},"selection_glyph":null,"view":{"id":"26817"}},"id":"26816","type":"GlyphRenderer"},{"attributes":{"line_dash":[6],"location":2.480769230769231},"id":"26818","type":"Span"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26720"},{"id":"26721"},{"id":"26722"},{"id":"26723"},{"id":"26724"},{"id":"26725"},{"id":"26726"},{"id":"26727"}]},"id":"26730","type":"Toolbar"},{"attributes":{"data":{"top":{"__ndarray__":"EPzAD/zACUAg+IEf+IELQMEP/MAPfApAdmIndmKnC0A4ciM3ciMOQIZe6IVeaA1Ah17ohV5oDUDYiZ3YiR0NQD7wAz/wAw9Ae6EXeqEXDEAbuZEbuRELQHZiJ3ZipwtAeqEXeqEXDEB0IzdyIzcLQBu5kRu5EQtAFDuxEzsxCkByIzdyIzcLQBu5kRu5EQtAxU7sxE7sCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26855"},"selection_policy":{"id":"26854"}},"id":"26819","type":"ColumnDataSource"},{"attributes":{"below":[{"id":"26712"}],"center":[{"id":"26715"},{"id":"26719"},{"id":"26778"},{"id":"26784"},{"id":"26790"},{"id":"26796"}],"left":[{"id":"26716"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26776"},{"id":"26782"},{"id":"26788"},{"id":"26794"}],"title":{"id":"26799"},"toolbar":{"id":"26730"},"toolbar_location":null,"x_range":{"id":"26704"},"x_scale":{"id":"26708"},"y_range":{"id":"26706"},"y_scale":{"id":"26710"}},"id":"26703","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"26723","type":"WheelZoomTool"},{"attributes":{"source":{"id":"26819"}},"id":"26823","type":"CDSView"},{"attributes":{"callback":null},"id":"26727","type":"HoverTool"},{"attributes":{"data_source":{"id":"26819"},"glyph":{"id":"26820"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26821"},"selection_glyph":null,"view":{"id":"26823"}},"id":"26822","type":"GlyphRenderer"},{"attributes":{},"id":"26848","type":"UnionRenderers"},{"attributes":{"toolbars":[{"id":"26730"},{"id":"26764"}],"tools":[{"id":"26720"},{"id":"26721"},{"id":"26722"},{"id":"26723"},{"id":"26724"},{"id":"26725"},{"id":"26726"},{"id":"26727"},{"id":"26754"},{"id":"26755"},{"id":"26756"},{"id":"26757"},{"id":"26758"},{"id":"26759"},{"id":"26760"},{"id":"26761"}]},"id":"26860","type":"ProxyToolbar"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26763","type":"PolyAnnotation"},{"attributes":{"line_dash":[6],"location":3.480769230769231},"id":"26824","type":"Span"},{"attributes":{},"id":"26849","type":"Selection"},{"attributes":{},"id":"26839","type":"UnionRenderers"},{"attributes":{},"id":"26840","type":"Selection"},{"attributes":{},"id":"26721","type":"PanTool"},{"attributes":{"text":"tau"},"id":"26799","type":"Title"},{"attributes":{},"id":"26706","type":"DataRange1d"},{"attributes":{"overlay":{"id":"26729"}},"id":"26724","type":"LassoSelectTool"},{"attributes":{},"id":"26850","type":"UnionRenderers"},{"attributes":{},"id":"26851","type":"Selection"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26729","type":"PolyAnnotation"},{"attributes":{},"id":"26760","type":"SaveTool"},{"attributes":{},"id":"26742","type":"LinearScale"},{"attributes":{},"id":"26726","type":"SaveTool"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26830"},"ticker":{"id":"26797"}},"id":"26716","type":"LinearAxis"},{"attributes":{},"id":"26713","type":"BasicTicker"},{"attributes":{},"id":"26710","type":"LinearScale"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26775","type":"VBar"},{"attributes":{"line_dash":[6],"location":0.41666666666666663},"id":"26778","type":"Span"},{"attributes":{},"id":"26852","type":"UnionRenderers"},{"attributes":{"source":{"id":"26773"}},"id":"26777","type":"CDSView"},{"attributes":{"axis":{"id":"26712"},"ticker":null},"id":"26715","type":"Grid"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26781","type":"VBar"},{"attributes":{},"id":"26853","type":"Selection"},{"attributes":{"data_source":{"id":"26773"},"glyph":{"id":"26774"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26775"},"selection_glyph":null,"view":{"id":"26777"}},"id":"26776","type":"GlyphRenderer"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26831"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26713"}},"id":"26712","type":"LinearAxis"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26786","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAA8D8OdNpApw30PxSuR+F6FPY/1AY6baDT9T8c6LSBThv4PxdLfrHkF/c/1QY6baDT9T+V/GLJL5b2P1jyiyW/WPc/43oUrkfh+T8ehetRuB75PxdLfrHkF/c/mJmZmZmZ9z8YrkfhehT2P1RVVVVVVfY/lfxiyS+W9j/gehSuR+H5P5iZmZmZmfc/kl8s+cWS9T8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26836"},"selection_policy":{"id":"26835"}},"id":"26779","type":"ColumnDataSource"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26780","type":"VBar"},{"attributes":{"source":{"id":"26779"}},"id":"26783","type":"CDSView"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26787","type":"VBar"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26792","type":"VBar"},{"attributes":{"data_source":{"id":"26779"},"glyph":{"id":"26780"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26781"},"selection_glyph":null,"view":{"id":"26783"}},"id":"26782","type":"GlyphRenderer"},{"attributes":{},"id":"26831","type":"BasicTickFormatter"},{"attributes":{"axis":{"id":"26746"},"ticker":null},"id":"26749","type":"Grid"},{"attributes":{"callback":null},"id":"26761","type":"HoverTool"},{"attributes":{"line_dash":[6],"location":1.4166666666666665},"id":"26784","type":"Span"},{"attributes":{},"id":"26854","type":"UnionRenderers"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26845"},"ticker":{"id":"26825"}},"id":"26750","type":"LinearAxis"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAAAECkcD0K1yMBQCa/WPKLpQFA6LSBThtoAkBqA5020OkCQCz5xZJfrANA8O7u7u5uBEDrUbgehWsDQOtRuB6FawNAC9ejcD0KA0BKfrHkF0sDQA102kCnDQRAqqqqqqoqA0BQG+i0gU4EQC+W/GLJrwRAThvotIFOBECuR+F6FC4EQE4b6LSBTgRAcD0K16PwBEA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26838"},"selection_policy":{"id":"26837"}},"id":"26785","type":"ColumnDataSource"},{"attributes":{},"id":"26845","type":"BasicTickFormatter"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26846"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26747"}},"id":"26746","type":"LinearAxis"},{"attributes":{"source":{"id":"26785"}},"id":"26789","type":"CDSView"},{"attributes":{},"id":"26855","type":"Selection"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26793","type":"VBar"},{"attributes":{},"id":"26747","type":"BasicTicker"},{"attributes":{},"id":"26725","type":"UndoTool"},{"attributes":{"line_dash":[6],"location":0.48076923076923067},"id":"26806","type":"Span"},{"attributes":{"data_source":{"id":"26785"},"glyph":{"id":"26786"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26787"},"selection_glyph":null,"view":{"id":"26789"}},"id":"26788","type":"GlyphRenderer"},{"attributes":{"ticks":[0,1,2,3]},"id":"26797","type":"FixedTicker"},{"attributes":{"line_dash":[6],"location":2.4166666666666665},"id":"26790","type":"Span"},{"attributes":{"axis":{"id":"26750"},"dimension":1,"ticker":null},"id":"26753","type":"Grid"},{"attributes":{"data":{"top":{"__ndarray__":"MzMzMzOzDUC4HoXrUTgPQDCW/GLJrwxA0GkDnTbQDEBtoNMGOu0LQOi0gU4baApAzszMzMzMC0DrUbgehWsLQClcj8L1qApAqA102kAnCkBH4XoUrkcKQMaSXyz5xQlA6LSBThtoCkAqXI/C9agKQClcj8L1qApACtejcD0KC0AGOm2g0wYKQMkvlvxiyQpAaQOdNtDpCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26840"},"selection_policy":{"id":"26839"}},"id":"26791","type":"ColumnDataSource"},{"attributes":{"data":{"top":{"__ndarray__":"ZWZmZmZm7j9OG+i0gU7XP2cDnTbQad8/WfKLJb9Y2j9Bpw102kDTP17JL5b8Yt0/PW2g0wY60T9U8oslv1jaP1ws+cWSX9w/WlVVVVVV2T9SVVVVVVXZPzTQaQOdNuA/ZgOdNtBp3z9m8oslv1jaP0h+seQXS9Y/SH6x5BdL1j84baDTBjrRPz+nDXTaQNM/SH6x5BdL1j8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26834"},"selection_policy":{"id":"26833"}},"id":"26773","type":"ColumnDataSource"},{"attributes":{"source":{"id":"26791"}},"id":"26795","type":"CDSView"},{"attributes":{},"id":"26833","type":"UnionRenderers"},{"attributes":{"overlay":{"id":"26762"}},"id":"26756","type":"BoxZoomTool"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26774","type":"VBar"},{"attributes":{},"id":"26755","type":"PanTool"},{"attributes":{"ticks":[0,1,2,3]},"id":"26825","type":"FixedTicker"},{"attributes":{},"id":"26720","type":"ResetTool"},{"attributes":{},"id":"26754","type":"ResetTool"},{"attributes":{"data_source":{"id":"26791"},"glyph":{"id":"26792"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26793"},"selection_glyph":null,"view":{"id":"26795"}},"id":"26794","type":"GlyphRenderer"},{"attributes":{},"id":"26834","type":"Selection"},{"attributes":{},"id":"26757","type":"WheelZoomTool"},{"attributes":{"line_dash":[6],"location":3.4166666666666665},"id":"26796","type":"Span"},{"attributes":{"overlay":{"id":"26763"}},"id":"26758","type":"LassoSelectTool"},{"attributes":{"source":{"id":"26801"}},"id":"26805","type":"CDSView"},{"attributes":{},"id":"26759","type":"UndoTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26754"},{"id":"26755"},{"id":"26756"},{"id":"26757"},{"id":"26758"},{"id":"26759"},{"id":"26760"},{"id":"26761"}]},"id":"26764","type":"Toolbar"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26803","type":"VBar"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26802","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"6YVe6IVe4D9nZmZmZmbeP2dmZmZmZu4/WWqlVmql7D/eyI3cyI3YP7vQC73QC9U/uBM7sRM73T+vEzuxEzvdPyZ2Yid2Ytc/lxu5kRu52T8ZuZEbuZHfP5AbuZEbudk/QS/0Qi/04D8LwQ/8wA/cP5AbuZEbudk/q9RKrdRK4z9BL/RCL/TgPyZ2Yid2Ytc/USu1Uiu1wj8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26849"},"selection_policy":{"id":"26848"}},"id":"26801","type":"ColumnDataSource"},{"attributes":{"overlay":{"id":"26728"}},"id":"26722","type":"BoxZoomTool"},{"attributes":{"text":"mu"},"id":"26827","type":"Title"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26809","type":"VBar"},{"attributes":{},"id":"26708","type":"LinearScale"},{"attributes":{"data_source":{"id":"26801"},"glyph":{"id":"26802"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26803"},"selection_glyph":null,"view":{"id":"26805"}},"id":"26804","type":"GlyphRenderer"},{"attributes":{},"id":"26835","type":"UnionRenderers"},{"attributes":{"children":[[{"id":"26703"},0,0],[{"id":"26739"},0,1]]},"id":"26859","type":"GridBox"},{"attributes":{"line_dash":[6],"location":1.4807692307692308},"id":"26812","type":"Span"},{"attributes":{"axis":{"id":"26716"},"dimension":1,"ticker":null},"id":"26719","type":"Grid"},{"attributes":{},"id":"26830","type":"BasicTickFormatter"},{"attributes":{"data":{"top":{"__ndarray__":"P/ADP/AD9z+SG7mRG7n2Py/0Qi/0QvU/eqEXeqEX9D9IbuRGbuT3P4If+IEf+PQ/MPRCL/RC9T+ZmZmZmZn3Pyd2Yid2YvQ/9kIv9EIv+D+4kRu5kRv7P7ATO7ETO/o/oBd6oRd6+D+mF3qhF3r4P1ZqpVZqpfk/9EIv9EIv+D9GbuRGbuT3P07sxE7sxPg/wA/8wA/8+z8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26851"},"selection_policy":{"id":"26850"}},"id":"26807","type":"ColumnDataSource"},{"attributes":{},"id":"26836","type":"Selection"},{"attributes":{"toolbar":{"id":"26860"},"toolbar_location":"above"},"id":"26861","type":"ToolbarBox"},{"attributes":{"below":[{"id":"26746"}],"center":[{"id":"26749"},{"id":"26753"},{"id":"26806"},{"id":"26812"},{"id":"26818"},{"id":"26824"}],"left":[{"id":"26750"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26804"},{"id":"26810"},{"id":"26816"},{"id":"26822"}],"title":{"id":"26827"},"toolbar":{"id":"26764"},"toolbar_location":null,"x_range":{"id":"26704"},"x_scale":{"id":"26742"},"y_range":{"id":"26706"},"y_scale":{"id":"26744"}},"id":"26739","subtype":"Figure","type":"Plot"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26808","type":"VBar"},{"attributes":{"source":{"id":"26807"}},"id":"26811","type":"CDSView"},{"attributes":{"data_source":{"id":"26807"},"glyph":{"id":"26808"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26809"},"selection_glyph":null,"view":{"id":"26811"}},"id":"26810","type":"GlyphRenderer"}],"root_ids":["26862"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"75e54a11-f7ee-472f-b73e-c156bd6aa06f","root_ids":["26862"],"roots":{"26862":"3a519593-e510-4232-9be0-017895eb94ba"}}];
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